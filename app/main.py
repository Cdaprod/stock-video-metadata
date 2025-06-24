# app/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import sys
import shutil
import json

# ── Locate the project root (one level above `app/`) ────────────────────────
# 1) Locate the project root (one level above `app/`)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 2) Add both the `scripts/` folder and the project root itself ─────────────
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

# ── core & facades ──────────────────────────────────────────────────────────
from VideoArtifact import (
    ArtifactFactory as BatchFactory,
    BatchProcessor,
    VideoArtifact,
)
from VideoFacade import VideoFacade          # if in /scripts

# ── import your legacy pipelines ────────────────────────────────────────────
from discovery import discover_video_batches, save_inventory
from enrichment  import VideoEnricher
from curation    import extract_audio, curate_clip

# ── Module(s) Imports ──────────────────────────────────────────────────────────
from app.modules.content_pipeline import router as content_router

# ── app + CORS ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Video Metadata Pipeline API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # adjust in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── directories for media + metadata ──────────────────────────────────────────
MEDIA_DIR    = PROJECT_ROOT / "media"
METADATA_DIR = PROJECT_ROOT / "metadata"
MEDIA_DIR.mkdir(exist_ok=True, parents=True)
METADATA_DIR.mkdir(exist_ok=True, parents=True)

# ── shared batch processor ─────────────────────────────────────────────────────
processor = BatchProcessor()


# ── helper Pydantic model for "from-paths" endpoint ────────────────────────────
class PathsPayload(BaseModel):
    paths: List[str]
    batch: Optional[str] = None

# ── manifest → proxy helper ───────────────────────────────────────────────────
def manifest_to_facade_proxies(manifest_path: Path):
    """
    Read a batch manifest JSON, rehydrate each VideoArtifact,
    wrap in VideoFacade, and return list of MediaProxyArtifact.
    """
    if not manifest_path.exists():
        return None
    batch_json = json.loads(manifest_path.read_text())
    proxies = [
        VideoFacade(artifact=VideoArtifact(**v)).to_proxy()
        for v in batch_json.get("videos", [])
    ]
    return {"batch_id": batch_json.get("id"), "videos": proxies}


# ── app.modules routers ────────────────────────────────────────────
app.include_router(content_router)


# ── 1️⃣ Legacy "upload single file" ────────────────────────────────────────────
@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    batch: str = Form("uploads")
):
    batch_dir = MEDIA_DIR / batch
    batch_dir.mkdir(exist_ok=True)
    dest = batch_dir / file.filename
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return {
        "status":   "uploaded",
        "batch":    batch,
        "filename": file.filename,
        "path":     str(dest)
    }


# ── 2️⃣ New: ingest a batch by upload (your HTML calls this) ──────────────────
@app.post("/batches/from-upload/")
async def web_batch_from_upload(
    files: List[UploadFile] = File(...),
    batch: str = Form("uploads")
):
    # 1) save to MEDIA_DIR
    uploads = []
    batch_dir = MEDIA_DIR / batch
    batch_dir.mkdir(exist_ok=True, parents=True)
    for f in files:
        dest = batch_dir / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        uploads.append({
            "filename": f.filename,
            "data":     dest.read_bytes()
        })

    # 2) create & process the batch artifact
    batch_artifact = BatchFactory.create_batch_from_api(
        uploads,
        config={},
        request_metadata={"source": "api", "batch": batch}
    )
    processor.process_batch(batch_artifact)

    # 3) persist manifest
    manifest_path = batch_artifact.save_manifest(output_dir=str(METADATA_DIR))
    return {
        "batch_id": batch_artifact.id,
        "manifest": manifest_path
    }


# ── 3️⃣ New: ingest a batch by server-side paths ───────────────────────────────
@app.post("/batches/from-paths/")
async def web_batch_from_paths(payload: PathsPayload = Body(...)):
    # emulate CLI ingestion
    fake_args = type("Args", (), {
        "batch_name": payload.batch,
        "quality":    "medium",
        "format":     "mp4",
        "output":     "./outputs"
    })()
    batch_artifact = BatchFactory.create_batch_from_cli(
        fake_args,
        payload.paths
    )
    processor.process_batch(batch_artifact)
    manifest_path = batch_artifact.save_manifest(output_dir=str(METADATA_DIR))
    return {
        "batch_id": batch_artifact.id,
        "manifest": manifest_path
    }


# ── 4️⃣ List all existing batches ──────────────────────────────────────────────
@app.get("/batches/")
def list_batches():
    manifests = list(METADATA_DIR.glob("*_manifest.json"))
    out = []
    if manifests:
        for m in manifests:
            try:
                j = json.loads(m.read_text())
                out.append({
                    "id":   j.get("id"),
                    "name": j.get("name", j.get("id"))
                })
            except:
                continue
    else:
        # fallback: list dirs under media/
        for d in MEDIA_DIR.iterdir():
            if d.is_dir():
                out.append({"id": d.name, "name": d.name})
    return {"batches": out}


# ── 5️⃣ Get one batch’s details ───────────────────────────────────────────────
@app.get("/batches/{batch_id}/")
def get_batch(batch_id: str):
    manifest = METADATA_DIR / f"{batch_id}_manifest.json"
    if manifest.exists():
        batch_json = json.loads(manifest.read_text())
        # Wrap each video artifact in a VideoFacade and output as proxy:
        videos = batch_json.get("videos", [])
        enhanced_videos = []
        for v in videos:
            # Rehydrate VideoArtifact if needed:
            va = VideoArtifact(**v)
            facade = VideoFacade(artifact=va)
            # You could register additional fields dynamically here!
            proxy_obj = facade.to_proxy()
            enhanced_videos.append(proxy_obj)
        # Return enhanced proxy objects in API:
        batch_json["videos"] = enhanced_videos
        return batch_json

    # fallback: just list files in media/<batch_id>/
    dirp = MEDIA_DIR / batch_id
    if dirp.exists() and dirp.is_dir():
        videos = []
        for vid in dirp.iterdir():
            videos.append({
                "filename": vid.name,
                "full_path": str(vid),
                "state": "uploaded"
            })
        return {"id": batch_id, "name": batch_id, "videos": videos}
    raise HTTPException(404, f"Batch {batch_id} not found")


# ── To let clients retrieve a proxy/facade view of any batch: ───────────────────
@app.get("/batches/{batch_id}/proxy/")
def get_batch_proxy(batch_id: str):
    manifest = METADATA_DIR / f"{batch_id}_manifest.json"
    data = manifest_to_facade_proxies(manifest)
    if data is None:
        raise HTTPException(404, f"Batch {batch_id} not found")
    return data

# ── 6️⃣ Legacy "discover / inventory" ─────────────────────────────────────────
@app.post("/discover/")
def discover_batches():
    batches = discover_video_batches()
    save_inventory(
      batches,
      out_json = METADATA_DIR / "batch_metadata.json",
      out_csv  = METADATA_DIR / "video_inventory.csv"
    )
    return {
      "batches": list(batches.keys()),
      "count":   len(batches)
    }


# ── 7️⃣ Legacy "enrich" ───────────────────────────────────────────────────────
@app.post("/enrich/")
def enrich_inventory():
    inv_csv = METADATA_DIR / "video_inventory.csv"
    out_csv = METADATA_DIR / "enriched_videos.csv"
    enricher = VideoEnricher()
    import pandas as pd
    df = pd.read_csv(inv_csv)
    df_enriched = enricher.enrich_dataframe(df, enriched_csv=str(out_csv))
    df_enriched.to_csv(out_csv, index=False)
    return {"status": "enriched", "output_csv": str(out_csv)}


# ── 8️⃣ Legacy "curate" ────────────────────────────────────────────────────────
@app.post("/curate/")
def curate_batch(batch: str = Form("uploads")):
    batch_dir = MEDIA_DIR / batch
    results = []
    for vid in batch_dir.iterdir():
        audio = extract_audio(str(vid))
        meta  = curate_clip(audio, str(vid), out_dir=str(METADATA_DIR/"curation"))
        results.append({"video": str(vid), **meta})
    return {"results": results}


# ── 9️⃣ Health Status "FastAPI " ────────────────────────────────────────────────────────
@app.get("/health/", response_model=dict)
def health_check():
    """
    Basic health check endpoint.
    Extend this to include checks for DB, S3, etc as needed.
    """
    status = {
        "status": "ok",
        "dependencies": {
            "minio": "unverified",
            "weaviate": "unverified",
            "database": "unverified",
        }
    }
    # Optionally, check MinIO, Weaviate, DB status here and set value to "ok" or "error"
    return JSONResponse(status_code=200, content=status)


# -- Docker/production entrypoint --
def start():
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    # This allows `python app/main.py` to work (local dev)
    start()