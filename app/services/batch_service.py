# app/services/batch_service.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.factory   import ArtifactFactory
from app.core.processor import BatchProcessor
from app.core.artifacts.video import VideoArtifact
from app.core.facades.video_facade import VideoFacade
from .base import ArtifactService

BASE_DIR  = Path(__file__).resolve().parent.parent
MEDIA_DIR = BASE_DIR / "media"
META_DIR  = BASE_DIR / "metadata"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

class BatchService(ArtifactService):
    """Fully autonomous BatchArtifact lifecycle, with legacy safety-net fallback."""
    def __init__(self) -> None:
        self._processor = BatchProcessor()

    def ingest_uploads(
        self,
        uploads: List[Dict[str, Any]],
        batch: str = "uploads",
        **kwargs: Any
    ) -> str:
        art = ArtifactFactory.create_batch_from_api(
            uploads,
            config={},
            request_metadata={"source": "api", "batch": batch}
        )
        self._processor.process_batch(art)
        art.save_manifest(output_dir=str(META_DIR))
        return art.id

    def ingest_paths(
        self,
        paths: List[str],
        batch_name: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        fake = type("Args", (), {
            "batch_name": batch_name,
            "quality":    "medium",
            "format":     "mp4",
            "output":     "./outputs"
        })()
        art = ArtifactFactory.create_batch_from_cli(fake, paths)
        self._processor.process_batch(art)
        art.save_manifest(output_dir=str(META_DIR))
        return art.id

    def list(self) -> List[Dict[str, str]]:
        manifests = list(META_DIR.glob("*_manifest.json"))
        out: List[Dict[str,str]] = []
        if manifests:
            for mp in manifests:
                try:
                    j = json.loads(mp.read_text())
                    out.append({"id": j["id"], "name": j.get("name", j["id"])})
                except Exception:
                    continue
        else:
            for d in MEDIA_DIR.iterdir():
                if d.is_dir():
                    out.append({"id": d.name, "name": d.name})
        return out

    def get(self, artifact_id: str) -> Dict[str, Any]:
        mp = META_DIR / f"{artifact_id}_manifest.json"
        if mp.exists():
            raw = json.loads(mp.read_text())
            proxies = [
                VideoFacade(artifact=VideoArtifact(**v)).to_proxy().dict()
                for v in raw.get("videos", [])
            ]
            raw["videos"] = proxies
            return raw

        fallback_dir = MEDIA_DIR / artifact_id
        if fallback_dir.is_dir():
            vids = [
                {"filename": v.name, "full_path": str(v), "state": "uploaded"}
                for v in fallback_dir.iterdir()
            ]
            return {"id": artifact_id, "name": artifact_id, "videos": vids}

        raise FileNotFoundError(f"Batch '{artifact_id}' not found")

    def manifest(self, artifact_id: str) -> Dict[str, Any]:
        mp = META_DIR / f"{artifact_id}_manifest.json"
        if not mp.exists():
            raise FileNotFoundError(f"Manifest for '{artifact_id}' not found")
        return json.loads(mp.read_text())