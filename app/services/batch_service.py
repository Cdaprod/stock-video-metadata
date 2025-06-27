# app/services/batch_service.py
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
from app.core.factory   import ArtifactFactory
from app.core.processor import BatchProcessor
from .base import ArtifactService

BASE_DIR   = Path(__file__).resolve().parent.parent
MEDIA_DIR  = BASE_DIR / "media" / "batches"
META_DIR   = BASE_DIR / "metadata"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

_processor = BatchProcessor()

class BatchService(ArtifactService):
    """Service for BatchArtifact lifecycles."""

    # ---------- ingest ----------
    def ingest_uploads(self, uploads: List[Dict[str, Any]], batch="uploads", **_) -> str:
        batch_art = ArtifactFactory.create_batch_from_api(
            uploads, config={}, request_metadata={"source": "api", "batch": batch}
        )
        _processor.process_batch(batch_art)
        batch_art.save_manifest(output_dir=str(META_DIR))
        return batch_art.id

    def ingest_paths(self, paths: List[str], batch_name=None, **_) -> str:
        fake = type("Args", (), {"batch_name": batch_name, "quality":"medium",
                                 "format":"mp4", "output":"./outputs"})()
        batch_art = ArtifactFactory.create_batch_from_cli(fake, paths)
        _processor.process_batch(batch_art)
        batch_art.save_manifest(output_dir=str(META_DIR))
        return batch_art.id

    # ---------- queries ----------
    def list(self):
        out = []
        for mp in META_DIR.glob("*_manifest.json"):
            j = json.loads(mp.read_text())
            out.append({"id": j["id"], "name": j.get("name", j["id"])})
        return out

    def get(self, artifact_id: str):
        return self.manifest(artifact_id)  # proxy already included there

    def manifest(self, artifact_id: str):
        mp = META_DIR / f"{artifact_id}_manifest.json"
        if not mp.exists():
            raise FileNotFoundError(artifact_id)
        return json.loads(mp.read_text())