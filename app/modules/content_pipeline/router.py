from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from typing import List, Dict, Any
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime

# Import your core logic (classes, functions) from content_pipeline.py
from .content_pipeline import (
    ContentExtractor,
    ScriptGenerator,
    ContentJobManager,
    PromptResponse,
    ContentScript,
    ContentJob,
    extractor,
    script_gen,
    job_manager
)

router = APIRouter(
    prefix="/content",
    tags=["content-pipeline"]
)

# === Models for request bodies (if not imported from content_pipeline.py) ===

# === Endpoints ===

@router.post("/extract/screenshot/", response_model=List[PromptResponse])
async def extract_from_screenshot(file: UploadFile = File(...)):
    """
    Upload a screenshot and extract prompt/response pairs via OCR.
    """
    temp_path = Path(f"/tmp/{datetime.now().timestamp()}_{file.filename}")
    try:
        with temp_path.open("wb") as f:
            content = await file.read()
            f.write(content)
        text = extractor.extract_from_screenshot(temp_path)
        pairs = extractor.identify_prompt_response_pairs(text)
        return pairs
    finally:
        if temp_path.exists():
            temp_path.unlink()

@router.post("/extract/video/", response_model=Dict[str, Any])
async def extract_from_video(file: UploadFile = File(...)):
    """
    Upload a video and extract content (OCR, audio, prompts).
    """
    temp_path = Path(f"/tmp/{datetime.now().timestamp()}_{file.filename}")
    try:
        with temp_path.open("wb") as f:
            content = await file.read()
            f.write(content)
        extraction_result = extractor.extract_from_video(temp_path)
        # Post-process: extract prompt-response pairs from OCR chunks if available
        if 'ocr_chunks' in extraction_result:
            all_text = ' '.join([chunk['text'] for chunk in extraction_result['ocr_chunks']])
            pairs = extractor.identify_prompt_response_pairs(all_text)
            extraction_result['prompt_response_pairs'] = [p.dict() for p in pairs]
        return extraction_result
    finally:
        if temp_path.exists():
            temp_path.unlink()

@router.post("/generate/script/", response_model=ContentScript)
async def generate_script(
    prompt_response: PromptResponse,
    format_type: str = Form("short")
):
    """
    Generate a video script from a prompt/response.
    """
    if format_type == "short":
        return script_gen.generate_short_script(prompt_response)
    else:
        return script_gen.generate_long_form_script([prompt_response])

@router.post("/create/auto-content/")
async def create_auto_content(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    format_type: str = Form("vertical"),
    style: str = Form("tutorial")
):
    """
    Automatically create content from uploaded screenshots/videos (async job).
    """
    file_names = [f.filename for f in files]
    job_id = job_manager.create_job(file_names, "auto-content")
    background_tasks.add_task(
        process_auto_content,
        job_id, files, format_type, style
    )
    return {"job_id": job_id, "status": "processing"}

# This needs to be imported from content_pipeline.py as well,
# but if it's not, define it here (with async def as you had):
from .content_pipeline import process_auto_content

@router.get("/jobs/{job_id}/", response_model=ContentJob)
async def get_job_status(job_id: str):
    """
    Get the status of a content creation job.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.get("/jobs/", response_model=List[ContentJob])
async def list_jobs():
    """
    List all content creation jobs.
    """
    return list(job_manager.jobs.values())

@router.post("/batches/{batch_id}/create-content/")
async def create_content_from_batch(
    batch_id: str,
    background_tasks: BackgroundTasks,
    format_type: str = Form("vertical")
):
    """
    Create content from an existing video batch (stub for integration).
    """
    job_id = job_manager.create_job([f"batch_{batch_id}"], "batch-content")
    # This is a stub. Extend to load batch, process, etc.
    return {"job_id": job_id, "message": "Content creation started for batch"}