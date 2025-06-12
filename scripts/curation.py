# /scripts/curation.py
import cv2, numpy as np
import subprocess

def analyze_clip_quality(path: str, start_frame: int, end_frame: int):
    cap = cv2.VideoCapture(path)
    scores = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    prev_gray = None
    for f in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        motion = ((gray - prev_gray)**2).mean() if prev_gray is not None else 0
        prev_gray = gray
        scores.append((f, sharp, motion))
    cap.release()
    return scores

def cut_best_segment(input_path: str, output_path: str, fps=30, length_sec=10):
    # very simplified: take from 1s to 1s+length
    start = 1
    subprocess.run([
        "ffmpeg","-y","-ss", str(start), "-i", input_path,
        "-t", str(length_sec), "-an","-c:v","copy", output_path
    ], check=True)
    return output_path

######

from pathlib import Path
import pandas as pd

def strip_audio(video_path: Path, output_path: Path):
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-c", "copy", "-an", str(output_path)
    ]
    subprocess.run(cmd, check=True)

def trim_video(video_path: Path, output_path: Path, start: float, duration: float):
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ss", str(start), "-t", str(duration),
        "-c", "copy", str(output_path)
    ]
    subprocess.run(cmd, check=True)

def batch_strip_audio(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        input_path = Path(row["full_path"])
        out_path = output_dir / input_path.name
        strip_audio(input_path, out_path)

def batch_trim(df: pd.DataFrame, output_dir: Path, trim_config: dict):
    """
    trim_config = {
        'video_filename.mp4': (start_time_in_sec, duration_in_sec),
        ...
    }
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        filename = row["filename"]
        if filename in trim_config:
            start, duration = trim_config[filename]
            input_path = Path(row["full_path"])
            out_path = output_dir / f"trimmed_{filename}"
            trim_video(input_path, out_path, start, duration)

def filter_by_size(df, min_mb=0, max_gb=50):
    return df[(df['size_bytes'] >= min_mb * 1024**2) & (df['size_bytes'] <= max_gb * 1024**3)]

def rename_files(df, pattern_func):
    for i, row in df.iterrows():
        new_name = pattern_func(row['filename'])
        old_path = Path(row['full_path'])
        new_path = old_path.with_name(new_name)
        old_path.rename(new_path)
        
"""
# Example Notebook Cells:
from scripts.processor import VideoBatchProcessor
from scripts.curation import batch_strip_audio, batch_trim

df = VideoBatchProcessor("batches/").get_flat_inventory()
batch_strip_audio(df, Path("curated/stripped_audio"))

trim_config = {
    "clip1.mp4": (3.5, 7.0),  # trim from 3.5s for 7s
}
batch_trim(df, Path("curated/trimmed"), trim_config)
"""