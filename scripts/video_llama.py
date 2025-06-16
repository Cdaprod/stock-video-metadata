# /scripts/video_llama.py

import os
from pathlib import Path
import torch
import cv2
from PIL import Image
import numpy as np
from typing import List, Dict, Optional, Union
import logging

# HuggingFace imports
from transformers import AutoProcessor, AutoModelForCausalLM

# CONFIG -- edit as needed for your environment
DEFAULT_MODEL_NAME = "microsoft/git-base-vatex"  # lightweight baseline; see notes below

def extract_video_frames(
    video_path: Union[str, Path],
    frame_rate: int = 1,
    max_frames: Optional[int] = 16,
) -> List[Image.Image]:
    """Extract frames from video at the specified rate and max_frames."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = int(fps / frame_rate) if frame_rate else 1
    frames = []
    frame_idxs = list(range(0, total_frames, interval))[:max_frames]
    for i in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to PIL Image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(img)
    cap.release()
    return frames

def generate_video_caption(
    video_path: Union[str, Path],
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Union[str, List[str]]]:
    """
    Generate a video caption using a HuggingFace video-to-text model.
    Returns a dict with at least 'description' and optionally 'tags'.
    """
    # 1. Extract frames
    frames = extract_video_frames(video_path, frame_rate=1, max_frames=16)
    if not frames:
        raise RuntimeError(f"No frames found in {video_path}")
    # 2. Load processor & model
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    # 3. Preprocess frames (most models take a list of PIL Images)
    inputs = processor(images=frames, return_tensors="pt").to(device)
    # 4. Generate caption (model-dependent interface)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return {
        "description": caption,
        "frames_used": len(frames),
        "model": model_name,
    }

def enrich_dataframe_with_video_llama(df, video_path_col="full_path", out_col="VideoLLaMA_Description", model_name=DEFAULT_MODEL_NAME):
    """
    Batch-enrich a DataFrame of video paths with transformer-based video descriptions.
    """
    results = []
    for idx, row in df.iterrows():
        video_path = row[video_path_col]
        if not Path(video_path).exists():
            results.append("")
            continue
        try:
            cap = generate_video_caption(video_path, model_name=model_name)
            results.append(cap["description"])
        except Exception as e:
            logging.warning(f"Failed for {video_path}: {e}")
            results.append("")
    df[out_col] = results
    return df

if __name__ == "__main__":
    # Example CLI/test run
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser(description="Video-LLaMA video-to-text batch processor")
    parser.add_argument("--csv", required=True, help="Input CSV with video paths")
    parser.add_argument("--out", required=True, help="Output CSV with descriptions")
    parser.add_argument("--col", default="full_path", help="Column with video file paths")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name")
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    df = enrich_dataframe_with_video_llama(df, video_path_col=args.col, model_name=args.model)
    df.to_csv(args.out, index=False)
    print(f"✅ Wrote VideoLLaMA descriptions to {args.out}")