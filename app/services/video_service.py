# app/services/video_service.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

from app.core.factory      import ArtifactFactory
from app.core.artifacts.video import VideoArtifact
from app.core.facades.video_facade import VideoFacade
from .base import ArtifactService

MEDIA_DIR = Path(__file__).resolve().parent.parent / "media" / "singles"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
MANIFESTS = MEDIA_DIR / "_manifests"
MANIFESTS.mkdir(exist_ok=True)

class VideoService(ArtifactService):
    """Fully autonomous single-video artifact handler."""

    def _save_manifest(self, art: VideoArtifact) -> None:
        path = MANIFESTS / f"{art.id}.json"
        path.write_text(json.dumps(art.dict(), indent=2))

    def ingest_uploads(
        self,
        uploads: List[Dict[str, Any]],
        **kwargs: Any
    ) -> str:
        up = uploads[0]  # only 1 file expected in this service
        art = ArtifactFactory.create_video_from_upload(
            up["filename"], up["data"], metadata=up.get("metadata")
        )
        self._save_manifest(art)
        return art.id

    def ingest_paths(
        self,
        paths: List[str],
        **kwargs: Any
    ) -> str:
        art = ArtifactFactory.create_video_from_path(paths[0])
        self._save_manifest(art)
        return art.id

    def list(self) -> List[Dict[str, str]]:
        return [{"id": m.stem, "name": m.stem} for m in MANIFESTS.glob("*.json")]

    def get(self, artifact_id: str) -> Dict[str, Any]:
        return self.manifest(artifact_id)["proxy"]

    def manifest(self, artifact_id: str) -> Dict[str, Any]:
        path = MANIFESTS / f"{artifact_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Video manifest '{artifact_id}' not found")
        data = json.loads(path.read_text())
        proxy = VideoFacade(artifact=VideoArtifact(**data)).to_proxy()
        return {"id": artifact_id, "proxy": proxy.dict()}