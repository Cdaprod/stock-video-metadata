# app/main.py

from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
import sys, os, shutil

# --- Setup, but DO NOT run processing here ---
REPO_ROOT = Path.cwd()
SCRIPTS_PATH = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_PATH))

from scripts.discovery import discover_video_batches, save_inventory
from scripts.enrichment import VideoEnricher
from scripts.curation import curate_clip
# ...other imports

app = FastAPI(title="Video Metadata Pipeline API")

MEDIA_DIR = REPO_ROOT / "media"
METADATA_DIR = REPO_ROOT / "metadata"
MEDIA_DIR.mkdir(exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...), batch: str = Form("uploads")):
    batch_dir = MEDIA_DIR / batch
    batch_dir.mkdir(exist_ok=True)
    dest = batch_dir / file.filename
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"status": "uploaded", "batch": batch, "filename": file.filename, "path": str(dest)}

@app.post("/discover/")
def discover_batches():
    """API-triggered: run batch discovery and save inventory."""
    batches = discover_video_batches()
    save_inventory(batches, out_json = METADATA_DIR / "batch_metadata.json", out_csv = METADATA_DIR / "video_inventory.csv")
    return {"batches": list(batches.keys()), "count": len(batches)}

@app.post("/enrich/")
def enrich_inventory():
    """API-triggered: run enrichment."""
    inv_csv = METADATA_DIR / "video_inventory.csv"
    out_csv = METADATA_DIR / "enriched_videos.csv"
    enricher = VideoEnricher()
    import pandas as pd
    df = pd.read_csv(inv_csv)
    df_enriched = enricher.enrich_dataframe(df, enriched_csv=str(out_csv))
    df_enriched.to_csv(out_csv, index=False)
    return {"status": "enriched", "output_csv": str(out_csv)}

@app.post("/curate/")
def curate_batch(batch: str = Form("uploads")):
    batch_dir = MEDIA_DIR / batch
    out = []
    for vid in batch_dir.iterdir():
        audio = extract_audio(str(vid))
        meta = curate_clip(audio, str(vid), out_dir=str(METADATA_DIR/"curation"))
        out.append({"video": str(vid), **meta})
    return {"results": out}
# ...more endpoints as needed (export, download metadata, service status, etc)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)