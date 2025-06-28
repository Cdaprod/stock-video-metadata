# app/main.py
from fastapi                 import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses       import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles     import StaticFiles
from pydantic                import BaseModel
from typing                  import List, Optiona
from pathlib                 import Path
import sys
import shutil
import json

# ── Locate the project root (one level above `app/`) ────────────────────────
# 1) Locate the project root (one level above `app/`)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 2) Add both the `scripts/` folder and the project root itself ─────────────
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

# ── Core & Facades ──────────────────────────────────────────────────────────
from app.core.artifacts.video      import VideoArtifact
from app.core.factory              import ArtifactFactory as BatchFactory
from app.core.processor            import BatchProcessor
from app.core.facades.video_facade import VideoFacade

# ── Internal Services (svc's) ──────────────────────────────────────────────────────────
from app.services.video_service import VideoService
from app.services.batch_service import BatchService

# ── Module(s) Imports ──────────────────────────────────────────────────────────
from app.modules.content_pipeline.router import router as content_router
from app.modules.enrich.router           import router as enrich_router
from app.modules.curate.router           import router as curate_router

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

# ── app.services svc's ────────────────────────────────────────────
video_svc = VideoService()
batch_svc = BatchService()

# ── app.modules routers ────────────────────────────────────────────
app.include_router(content_router)
app.include_router(enrich_router)
app.include_router(curate_router)


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


# ── #️⃣ Mounted Static HTML for root path ────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "public" / "static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = PROJECT_ROOT / "public" / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text())
    
    
# ── 1️⃣ Upload ONE video (legacy path, now uses VideoService) ────────────────
@app.post("/upload/", response_model=dict)
async def upload_file(file: UploadFile = File(...)):
    """
    Accept one video and create a standalone VideoArtifact.
    """
    data = await file.read()                      # bytes in memory
    vid_id = video_svc.ingest_uploads(
        uploads=[{"filename": file.filename, "data": data}]
    )
    return {"video_id": vid_id, "filename": file.filename} 


# ── 2️⃣ Upload a BATCH of videos (HTML form calls this) ──────────────────────
@app.post("/batches/from-upload/", response_model=dict)
async def upload_batch(
    files: List[UploadFile] = File(...),
    batch: str = Form("uploads")
):
    """
    Accept multiple video files, build a BatchArtifact,
    process it, and persist the manifest.
    """
    uploads = []
    for f in files:
        uploads.append({
            "filename": f.filename,
            "data":     await f.read()
        })

    batch_id = batch_svc.ingest_uploads(uploads, batch=batch)
    return {"batch_id": batch_id}
    
    
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
      "manifest": str(manifest_path)
    }


# ── 4️⃣ List all existing batches ────────────────────────────────────────────
@app.get("/batches/", response_model=dict)
def list_batches():
    """
    Return lite info (id, name) for every saved batch manifest.
    """
    return {"batches": batch_service.list()}
    

# ── 5️⃣ Get one batch’s details (videos + proxy fields) ──────────────────────
@app.get("/batches/{batch_id}/", response_model=dict)
def get_batch(batch_id: str):
    """
    Return the proxy-enhanced manifest for a single batch.
    """
    try:
        return batch_svc.get(batch_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Batch {batch_id} not found")
         

# ── 6️⃣ Proxy / raw manifest view ────────────────────────────────────────────────
@app.get("/batches/{batch_id}/proxy/", response_model=dict)
def get_batch_manifest(batch_id: str):
    """
    Return the original manifest (no proxy transformation).
    """
    try:
        return batch_svc.manifest(batch_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Batch {batch_id} not found")
        
        
# # ── 6️⃣ Legacy "discover / inventory" ─────────────────────────────────────────
# @app.post("/discover/")
# def discover_batches():
#     batches = discover_video_batches()
#     save_inventory(
#       batches,
#       out_json = METADATA_DIR / "batch_metadata.json",
#       out_csv  = METADATA_DIR / "video_inventory.csv"
#     )
#     return {
#       "batches": list(batches.keys()),
#       "count":   len(batches)
#     }


# # ── 7️⃣ Legacy "enrich" ───────────────────────────────────────────────────────
# @app.post("/enrich/")
# def enrich_inventory():
#     inv_csv = METADATA_DIR / "video_inventory.csv"
#     out_csv = METADATA_DIR / "enriched_videos.csv"
#     enricher = VideoEnricher()
#     import pandas as pd
#     df = pd.read_csv(inv_csv)
#     df_enriched = enricher.enrich_dataframe(df, enriched_csv=str(out_csv))
#     df_enriched.to_csv(out_csv, index=False)
#     return {"status": "enriched", "output_csv": str(out_csv)}


# # ── 8️⃣ Legacy "curate" ────────────────────────────────────────────────────────
# @app.post("/curate/")
# def curate_batch(batch: str = Form("uploads")):
#     batch_dir = MEDIA_DIR / batch
#     results = []
#     for vid in batch_dir.iterdir():
#         audio = extract_audio(str(vid))
#         meta  = curate_clip(audio, str(vid), out_dir=str(METADATA_DIR/"curation"))
#         results.append({"video": str(vid), **meta})
#     return {"results": results}


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