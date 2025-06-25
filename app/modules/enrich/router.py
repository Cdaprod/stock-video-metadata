# app/modules/enrich/router.py
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd

from .enrichment_pipeline import VideoEnricher

router = APIRouter(prefix="/enrich", tags=["enrichment"])
_enricher = VideoEnricher()

@router.post("/single/", response_model=Dict[str, Any])
async def enrich_single(file: UploadFile = File(...)):
    # 1) save temp file
    tmp = Path("/tmp") / file.filename
    with tmp.open("wb") as f:
        f.write(await file.read())
    # 2) build a single-row DataFrame
    df = pd.DataFrame([{"filename": file.filename, "full_path": str(tmp), "batch_name": "api"}])
    df_enriched = _enricher.enrich_dataframe(df)
    return df_enriched.to_dict(orient="records")[0]

@router.post("/batch-from-csv/", response_model=List[Dict[str, Any]])
async def enrich_batch_from_csv(csv_path: str):
    # Expects a path to a CSV with columns full_path, batch_name, filename
    df = pd.read_csv(csv_path)
    df_out = _enricher.enrich_dataframe(df)
    # optional: save to new CSV & return path
    return df_out.to_dict(orient="records")