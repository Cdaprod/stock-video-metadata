# app/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import sys
import shutil
import json

# ── allow importing your scripts/ folder ───────────────────────────────────────
REPO_ROOT   = Path(__file__).parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# ── import the artifact‐based pipeline ────────────────────────────────────────
from VideoArtifact import ArtifactFactory as BatchFactory, BatchProcessor

# ── import your legacy pipelines ───────────────────────────────────────────────
from discovery import discover_video_batches, save_inventory
from enrichment  import VideoEnricher
from curation    import extract_audio, curate_clip

# ── app + CORS ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Video Metadata Pipeline API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # adjust in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── directories for media + metadata ──────────────────────────────────────────
MEDIA_DIR    = REPO_ROOT / "media"
METADATA_DIR = REPO_ROOT / "metadata"
MEDIA_DIR.mkdir(exist_ok=True, parents=True)
METADATA_DIR.mkdir(exist_ok=True, parents=True)

# ── shared batch processor ─────────────────────────────────────────────────────
processor = BatchProcessor()

# ── helper Pydantic model for "from-paths" endpoint ────────────────────────────
class PathsPayload(BaseModel):
    paths: List[str]
    batch: Optional[str] = None


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
        return json.loads(manifest.read_text())
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


# -- Docker/production entrypoint --
def start():
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    # This allows `python app/main.py` to work (local dev)
    start()