# /app/modules/curate/router.py

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Optional
from pathlib import Path
import shutil
import os
import json

# Import curation functions
from app.modules.curate.curation_pipeline import (
    curate_clip,
    batch_strip_audio,
    batch_trim,
    filter_by_size,
    rename_files,
)

router = APIRouter(
    prefix="/curate",
    tags=["curate"],
    responses={404: {"description": "Not found"}},
)

# Base directories (you can adjust as needed)
CURATED_DIR = Path(os.getenv("CURATED_DIR", "curated"))
CURATED_DIR.mkdir(parents=True, exist_ok=True)

# 1️⃣ Upload and curate a single video file
@router.post("/upload-and-curate/")
async def upload_and_curate(
    file: UploadFile = File(...),
    out_dir: Optional[str] = Form("curated")
):
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    dest = out_dir_path / file.filename

    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    meta = curate_clip(str(dest), str(out_dir_path))
    return {"filename": file.filename, "curation_metadata": meta}

# 2️⃣ Curate an existing video by path
@router.post("/curate-path/")
async def curate_path(payload: dict = Body(...)):
    video_path = payload.get("video_path")
    out_dir = payload.get("out_dir", "curated")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(404, f"Video not found: {video_path}")
    meta = curate_clip(video_path, out_dir)
    return {"video_path": video_path, "curation_metadata": meta}

# 3️⃣ Curate a batch of uploaded files
@router.post("/batch-upload-and-curate/")
async def batch_upload_and_curate(
    files: List[UploadFile] = File(...),
    out_dir: Optional[str] = Form("curated")
):
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    metadatas = []
    for f in files:
        dest = out_dir_path / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        meta = curate_clip(str(dest), str(out_dir_path))
        metadatas.append({"filename": f.filename, "curation_metadata": meta})
    return {"batch_results": metadatas}

# 4️⃣ List all curated files in a directory
@router.get("/list/")
async def list_curated_files(dir: Optional[str] = None):
    target_dir = Path(dir or CURATED_DIR)
    if not target_dir.exists():
        raise HTTPException(404, f"Directory not found: {target_dir}")
    files = [str(f) for f in target_dir.glob("*.json")]
    return {"curated_files": files}

# 5️⃣ Fetch curation metadata for a video file
@router.get("/get-metadata/")
async def get_metadata(filename: str, dir: Optional[str] = None):
    target_dir = Path(dir or CURATED_DIR)
    meta_path = target_dir / f"{Path(filename).stem}.json"
    if not meta_path.exists():
        raise HTTPException(404, f"Metadata not found for: {filename}")
    with meta_path.open() as f:
        meta = json.load(f)
    return meta

# 6️⃣ Batch operations (strip audio, trim, filter, rename)
@router.post("/batch-strip-audio/")
async def api_batch_strip_audio(payload: dict = Body(...)):
    # Expects {"df_csv": path_to_csv, "output_dir": str}
    import pandas as pd
    df_csv = payload["df_csv"]
    output_dir = Path(payload.get("output_dir", "curated/stripped_audio"))
    df = pd.read_csv(df_csv)
    batch_strip_audio(df, output_dir)
    return {"status": "audio stripped", "output_dir": str(output_dir)}

@router.post("/batch-trim/")
async def api_batch_trim(payload: dict = Body(...)):
    # Expects {"df_csv": path_to_csv, "output_dir": str, "trim_config": dict}
    import pandas as pd
    df_csv = payload["df_csv"]
    output_dir = Path(payload.get("output_dir", "curated/trimmed"))
    trim_config = payload["trim_config"]
    df = pd.read_csv(df_csv)
    batch_trim(df, output_dir, trim_config)
    return {"status": "batch trimmed", "output_dir": str(output_dir)}

# Add more as needed for your pipeline