# app/modules/content_pipeline/content_pipeline.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import cv2
import pytesseract
import subprocess
import tempfile
from datetime import datetime

# Import your existing components
from VideoFacade import VideoFacade
from VideoArtifact import VideoArtifact

router = APIRouter(prefix="/content", tags=["content-creation"])

# ── Need to work out prompt from frame text extraction ───────────────────────

# ── Models for Content Pipeline ──────────────────────────────────────────────
class PromptResponse(BaseModel):
    prompt: str
    response: str
    timestamp: datetime
    source_file: Optional[str] = None

class ContentScript(BaseModel):
    hook: str
    body: List[str]  # 3 bullet points
    cta: str
    estimated_duration: int  # seconds
    style: str = "tutorial"  # tutorial, demo, explanation

class VideoMetadata(BaseModel):
    title: str
    description: str
    hashtags: List[str]
    chapters: List[Dict[str, Any]]
    thumbnail_timestamp: Optional[float] = None

class ContentJob(BaseModel):
    id: str
    status: str  # pending, processing, completed, failed
    input_files: List[str]
    output_files: List[str]
    metadata: Dict[str, Any]
    created_at: datetime

# ── OCR and Content Extraction ──────────────────────────────────────────────
class ContentExtractor:
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def extract_from_screenshot(self, image_path: Path) -> str:
        """Extract text from screenshot using OCR"""
        try:
            import cv2
            import pytesseract
            
            image = cv2.imread(str(image_path))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            return text.strip()
        except Exception as e:
            return f"OCR extraction failed: {str(e)}"
    
    def extract_from_video(self, video_path: Path, sample_rate: int = 30) -> Dict[str, Any]:
        """Extract frames and audio from video for content analysis"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            ocr_chunks = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample every N frames based on fps
                if frame_count % int(fps * sample_rate) == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    text = pytesseract.image_to_string(gray)
                    if text.strip():
                        ocr_chunks.append({
                            "timestamp": frame_count / fps,
                            "text": text.strip()
                        })
                
                frame_count += 1
            
            cap.release()
            
            # Extract audio for potential STT
            audio_path = self.temp_dir / f"{video_path.stem}_audio.wav"
            subprocess.run([
                "ffmpeg", "-i", str(video_path), 
                "-vn", "-acodec", "pcm_s16le", 
                "-ar", "16000", "-ac", "1", 
                str(audio_path)
            ], capture_output=True)
            
            return {
                "ocr_chunks": ocr_chunks,
                "audio_path": str(audio_path) if audio_path.exists() else None,
                "duration": total_frames / fps,
                "fps": fps
            }
        except Exception as e:
            return {"error": str(e)}
    
    def identify_prompt_response_pairs(self, text: str) -> List[PromptResponse]:
        """Heuristically identify prompt/response patterns in extracted text"""
        lines = text.split('\n')
        pairs = []
        current_prompt = None
        current_response = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect prompts (customize these patterns for your use case)
            if (line.startswith('>') or 
                line.startswith('$') or 
                line.startswith('How') or 
                line.startswith('What') or 
                line.startswith('Can you')):
                
                if current_prompt and current_response:
                    pairs.append(PromptResponse(
                        prompt=current_prompt,
                        response=' '.join(current_response),
                        timestamp=datetime.now()
                    ))
                
                current_prompt = line
                current_response = []
            else:
                if current_prompt:
                    current_response.append(line)
        
        # Don't forget the last pair
        if current_prompt and current_response:
            pairs.append(PromptResponse(
                prompt=current_prompt,
                response=' '.join(current_response),
                timestamp=datetime.now()
            ))
        
        return pairs

# ── Content Script Generator ─────────────────────────────────────────────────
class ScriptGenerator:
    def __init__(self):
        # In production, you'd use your preferred LLM client
        # For now, this is a template-based approach
        pass
    
    def generate_short_script(self, prompt_response: PromptResponse) -> ContentScript:
        """Generate a 60-second vertical video script from prompt/response pair"""
        
        # This would normally call your LLM
        # For demo purposes, using template logic
        
        # Extract key concepts for hook
        prompt_words = prompt_response.prompt.split()[:5]
        hook = f"Quick tip: {' '.join(prompt_words)}"
        
        # Break response into digestible bullets
        response_sentences = prompt_response.response.split('.')[:3]
        body = [sentence.strip() + "." for sentence in response_sentences if sentence.strip()]
        
        # Ensure we have exactly 3 bullets
        while len(body) < 3:
            body.append("More details in the description below.")
        body = body[:3]
        
        cta = "Link in bio for the full guide 🔗"
        
        return ContentScript(
            hook=hook,
            body=body,
            cta=cta,
            estimated_duration=60,
            style="tutorial"
        )
    
    def generate_long_form_script(self, prompt_responses: List[PromptResponse]) -> ContentScript:
        """Generate a longer tutorial script from multiple prompt/response pairs"""
        
        combined_prompts = " | ".join([pr.prompt for pr in prompt_responses])
        hook = f"Deep dive: {combined_prompts[:50]}..."
        
        body = []
        for i, pr in enumerate(prompt_responses[:5]):  # Max 5 sections
            body.append(f"Section {i+1}: {pr.response[:100]}...")
        
        cta = "Subscribe for more technical tutorials!"
        
        return ContentScript(
            hook=hook,
            body=body,
            cta=cta,
            estimated_duration=len(prompt_responses) * 120,  # ~2 min per section
            style="deep-dive"
        )

# ── Video Production Pipeline ────────────────────────────────────────────────
class VideoProducer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def create_3d_screen_tilt(self, input_video: Path, output_video: Path):
        """Apply 3D camera tilt effect using FFmpeg"""
        try:
            subprocess.run([
                "ffmpeg", "-i", str(input_video),
                "-vf", "perspective=x0=0:y0=0*H:x1=W:y1=0.1*H:x2=0:y2=H:x3=W:y3=0.9*H",
                "-c:v", "libx264", "-preset", "fast",
                str(output_video)
            ], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"3D tilt failed: {e}")
            return False
    
    def generate_captions(self, script: ContentScript, duration: int) -> str:
        """Generate SRT captions from script"""
        srt_content = []
        current_time = 0
        
        # Hook timing (first 5 seconds)
        srt_content.append(f"1\n00:00:00,000 --> 00:00:05,000\n{script.hook}\n")
        current_time = 5
        
        # Body timing (split remaining time)
        body_duration = duration - 10  # Reserve 5s for hook, 5s for CTA
        section_duration = body_duration // len(script.body)
        
        for i, bullet in enumerate(script.body, 2):
            start_time = current_time
            end_time = current_time + section_duration
            
            start_srt = self._seconds_to_srt_time(start_time)
            end_srt = self._seconds_to_srt_time(end_time)
            
            srt_content.append(f"{i}\n{start_srt} --> {end_srt}\n{bullet}\n")
            current_time = end_time
        
        # CTA timing (last 5 seconds)
        start_srt = self._seconds_to_srt_time(current_time)
        end_srt = self._seconds_to_srt_time(duration)
        srt_content.append(f"{len(script.body)+2}\n{start_srt} --> {end_srt}\n{script.cta}\n")
        
        return "\n".join(srt_content)
    
    def _seconds_to_srt_time(self, seconds: int) -> str:
        """Convert seconds to SRT timestamp format"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d},000"
    
    def composite_final_video(self, 
                            screen_video: Path, 
                            script: ContentScript,
                            output_path: Path,
                            format_type: str = "vertical") -> bool:
        """Composite final video with captions and effects"""
        try:
            # Generate caption file
            srt_path = output_path.with_suffix('.srt')
            srt_content = self.generate_captions(script, script.estimated_duration)
            srt_path.write_text(srt_content)
            
            # Video dimensions based on format
            if format_type == "vertical":
                scale_filter = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
            else:
                scale_filter = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
            
            # FFmpeg command for final composition
            cmd = [
                "ffmpeg", "-i", str(screen_video),
                "-vf", f"{scale_filter},subtitles={srt_path}:force_style='Fontsize=24,PrimaryColour=&Hffffff,OutlineColour=&H000000'",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-t", str(script.estimated_duration),
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Video composition failed: {e}")
            return False

# ── Content Job Manager ──────────────────────────────────────────────────────
class ContentJobManager:
    def __init__(self, jobs_dir: Path):
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(exist_ok=True, parents=True)
        self.jobs = {}  # In production, use Redis or DB
    
    def create_job(self, input_files: List[str], job_type: str = "auto-content") -> str:
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        job = ContentJob(
            id=job_id,
            status="pending",
            input_files=input_files,
            output_files=[],
            metadata={"type": job_type},
            created_at=datetime.now()
        )
        self.jobs[job_id] = job
        return job_id
    
    def update_job_status(self, job_id: str, status: str, output_files: List[str] = None):
        if job_id in self.jobs:
            self.jobs[job_id].status = status
            if output_files:
                self.jobs[job_id].output_files = output_files
    
    def get_job(self, job_id: str) -> Optional[ContentJob]:
        return self.jobs.get(job_id)

# ── Initialize Components ────────────────────────────────────────────────────
extractor = ContentExtractor()
script_gen = ScriptGenerator()
job_manager = ContentJobManager(Path("./jobs"))

# ── API Endpoints ────────────────────────────────────────────────────────────

@router.post("/extract/screenshot/", response_model=List[PromptResponse])
async def extract_from_screenshot(file: UploadFile = File(...)):
    """Extract prompt/response pairs from screenshot"""
    temp_path = Path(tempfile.mktemp(suffix='.png'))
    
    try:
        with temp_path.open('wb') as f:
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
    """Extract content from screen recording"""
    temp_path = Path(tempfile.mktemp(suffix='.mp4'))
    
    try:
        with temp_path.open('wb') as f:
            content = await file.read()
            f.write(content)
        
        extraction_result = extractor.extract_from_video(temp_path)
        
        # Convert OCR chunks to prompt/response pairs
        if 'ocr_chunks' in extraction_result:
            all_text = ' '.join([chunk['text'] for chunk in extraction_result['ocr_chunks']])
            pairs = extractor.identify_prompt_response_pairs(all_text)
            extraction_result['prompt_response_pairs'] = [p.dict() for p in pairs]
        
        return extraction_result
    finally:
        if temp_path.exists():
            temp_path.unlink()

@router.post("/generate/script/", response_model=ContentScript)
async def generate_script(prompt_response: PromptResponse, format_type: str = "short"):
    """Generate video script from prompt/response pair"""
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
    """Automatically create content from uploaded screenshots/videos"""
    
    # Create job
    file_names = [f.filename for f in files]
    job_id = job_manager.create_job(file_names, "auto-content")
    
    # Process in background
    background_tasks.add_task(
        process_auto_content,
        job_id, files, format_type, style
    )
    
    return {"job_id": job_id, "status": "processing"}

async def process_auto_content(job_id: str, files: List[UploadFile], format_type: str, style: str):
    """Background task to process auto content creation"""
    try:
        job_manager.update_job_status(job_id, "processing")
        
        all_pairs = []
        temp_files = []
        
        # Extract content from all files
        for file in files:
            temp_path = Path(tempfile.mktemp(suffix=Path(file.filename).suffix))
            temp_files.append(temp_path)
            
            with temp_path.open('wb') as f:
                content = await file.read()
                f.write(content)
            
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                text = extractor.extract_from_screenshot(temp_path)
                pairs = extractor.identify_prompt_response_pairs(text)
                all_pairs.extend(pairs)
            elif file.filename.lower().endswith(('.mp4', '.mov', '.avi')):
                result = extractor.extract_from_video(temp_path)
                if 'ocr_chunks' in result:
                    all_text = ' '.join([chunk['text'] for chunk in result['ocr_chunks']])
                    pairs = extractor.identify_prompt_response_pairs(all_text)
                    all_pairs.extend(pairs)
        
        if not all_pairs:
            job_manager.update_job_status(job_id, "failed")
            return
        
        # Generate script
        if len(all_pairs) == 1:
            script = script_gen.generate_short_script(all_pairs[0])
        else:
            script = script_gen.generate_long_form_script(all_pairs)
        
        # For now, just save the script - in production you'd create the actual video
        output_dir = Path("./outputs") / job_id
        output_dir.mkdir(exist_ok=True, parents=True)
        
        script_path = output_dir / "script.json"
        script_path.write_text(script.json())
        
        job_manager.update_job_status(job_id, "completed", [str(script_path)])
        
    except Exception as e:
        job_manager.update_job_status(job_id, "failed")
        print(f"Auto content processing failed: {e}")
    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()

@router.get("/jobs/{job_id}/", response_model=ContentJob)
async def get_job_status(job_id: str):
    """Get content creation job status"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.get("/jobs/", response_model=List[ContentJob])
async def list_jobs():
    """List all content creation jobs"""
    return list(job_manager.jobs.values())

# ── Integration with existing batch system ──────────────────────────────────

@router.post("/batches/{batch_id}/create-content/")
async def create_content_from_batch(
    batch_id: str,
    background_tasks: BackgroundTasks,
    format_type: str = Form("vertical")
):
    """Create content from an existing video batch"""
    
    # This would integrate with your existing batch system
    # Get batch videos and process them through the content pipeline
    
    job_id = job_manager.create_job([f"batch_{batch_id}"], "batch-content")
    
    # In a real implementation, you'd:
    # 1. Load the batch from your existing system
    # 2. Extract frames/audio from batch videos
    # 3. Run through content pipeline
    # 4. Generate multiple pieces of content per batch
    
    return {"job_id": job_id, "message": "Content creation started for batch"}