# app/services/batch_service.py
from pathlib import Path
import json
from typing import List, Dict, Any, Optional

from app.core.factory   import ArtifactFactory
from app.core.processor import BatchProcessor

BASE_DIR   = Path(__file__).resolve().parent.parent
MEDIA_DIR  = BASE_DIR / "media"
META_DIR   = BASE_DIR / "metadata"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

_processor = BatchProcessor()

class BatchService:
    """Service for BatchArtifact lifecycles, with legacy-style fallbacks."""

    def ingest_uploads(self, uploads: List[Dict[str, Any]], batch: str="uploads") -> str:
        batch_art = ArtifactFactory.create_batch_from_api(
            uploads, config={}, request_metadata={"source": "api", "batch": batch}
        )
        _processor.process_batch(batch_art)
        batch_art.save_manifest(output_dir=str(META_DIR))
        return batch_art.id

    def ingest_paths(self, paths: List[str], batch_name: Optional[str]=None) -> str:
        fake = type("Args", (), {
            "batch_name": batch_name,
            "quality": "medium",
            "format": "mp4",
            "output": "./outputs"
        })()
        batch_art = ArtifactFactory.create_batch_from_cli(fake, paths)
        _processor.process_batch(batch_art)
        batch_art.save_manifest(output_dir=str(META_DIR))
        return batch_art.id

    def list(self) -> List[Dict[str, str]]:
        """
        1) Look for *_manifest.json files.
        2) If none, fall back to listing subdirs of MEDIA_DIR.
        """
        out: List[Dict[str,str]] = []
        for mp in META_DIR.glob("*_manifest.json"):
            try:
                j = json.loads(mp.read_text())
                out.append({"id": j["id"], "name": j.get("name", j["id"])})
            except Exception:
                continue

        if not out:
            # fallback: directories under media/
            for d in MEDIA_DIR.iterdir():
                if d.is_dir():
                    out.append({"id": d.name, "name": d.name})
        return out

    def get(self, batch_id: str) -> Dict[str, Any]:
        """
        1) If manifest exists, return the full proxy-enhanced manifest.
        2) Else if media/<batch_id> exists, list its files with minimal info.
        3) Else raise FileNotFoundError.
        """
        manifest_path = META_DIR / f"{batch_id}_manifest.json"
        if manifest_path.exists():
            # proxy-enhanced
            raw = json.loads(manifest_path.read_text())
            from app.core.facades.video_facade import VideoFacade
            from app.core.artifacts.video    import VideoArtifact

            proxies = []
            for v in raw.get("videos", []):
                va = VideoArtifact(**v)
                proxies.append(VideoFacade(artifact=va).to_proxy().dict())
            raw["videos"] = proxies
            return raw

        # fallback: list raw files in media/<batch_id>/
        dirp = MEDIA_DIR / batch_id
        if dirp.is_dir():
            videos = []
            for vid in dirp.iterdir():
                videos.append({
                    "filename": vid.name,
                    "full_path": str(vid),
                    "state": "uploaded"
                })
            return {"id": batch_id, "name": batch_id, "videos": videos}

        raise FileNotFoundError(batch_id)

    def manifest(self, batch_id: str) -> Dict[str, Any]:
        """
        Return the raw JSON manifest (no proxy).
        Strict: 404 if missing.
        """
        mp = META_DIR / f"{batch_id}_manifest.json"
        if not mp.exists():
            raise FileNotFoundError(batch_id)
        return json.loads(mp.read_text())