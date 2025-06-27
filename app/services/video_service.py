# app/services/video_service.py
from pathlib import Path
import json, uuid
from typing import Dict, Any, List, Optional
from app.core.factory import ArtifactFactory
from app.core.artifacts.video import VideoArtifact
from app.core.facades.video_facade import VideoFacade
from .base import ArtifactService

MEDIA_DIR = Path(__file__).resolve().parent.parent / "media" / "singles"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
MANIFESTS = MEDIA_DIR / "_manifests"; MANIFESTS.mkdir(exist_ok=True)

class VideoService(ArtifactService):
    """Service for single-video artifacts (legacy / ad-hoc uploads)."""

    def _save_manifest(self, art: VideoArtifact):
        mp = MANIFESTS / f"{art.id}.json"
        mp.write_text(json.dumps(art.dict(), indent=2))

    # ---------- interface impl ----------
    def ingest_uploads(self, uploads: List[Dict[str, Any]], **kw) -> str:
        up = uploads[0]                                   # one file expected
        art = ArtifactFactory.create_video_from_upload(
            up["filename"], up["data"], metadata=kw.get("metadata")
        )
        self._save_manifest(art)
        return art.id

    def ingest_paths(self, paths: List[str], **kw) -> str:
        art = ArtifactFactory.create_video_from_path(paths[0])
        self._save_manifest(art)
        return art.id

    def list(self):  # → [ {id, filename} ]
        return [{"id": p.stem, "name": p.stem}
                for p in MANIFESTS.glob("*.json")]

    def get(self, artifact_id: str):
        return self.manifest(artifact_id)["proxy"]

    def manifest(self, artifact_id: str):
        mp = MANIFESTS / f"{artifact_id}.json"
        if not mp.exists():
            raise FileNotFoundError(artifact_id)
        data = json.loads(mp.read_text())
        proxy = VideoFacade(artifact=VideoArtifact(**data)).to_proxy()
        return {"id": artifact_id, "proxy": proxy}

