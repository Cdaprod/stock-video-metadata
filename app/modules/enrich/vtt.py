# scripts/vtt.py

import cv2
import torch
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from typing import List, Optional

class VideoToText:
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        fallback: str   = "Salesforce/blip-image-captioning-base",
        max_frames: int = 5,
    ):
        """
        - model_name: the heavy BLIP2 model for best quality.
        - fallback:  a smaller BLIP for machines without BLIP2.
        - max_frames: how many evenly-spaced frames to caption.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_frames = max_frames

        # try BLIP2 pipeline first
        try:
            self.pipe = pipeline(
                "image-to-text",
                model=model_name,
                device=0 if self.device=="cuda" else -1,
                use_safetensors=True,
            )
        except Exception:
            # fallback to BLIP-base
            self.pipe = None
            self.processor = BlipProcessor.from_pretrained(fallback)
            self.model     = BlipForConditionalGeneration.from_pretrained(fallback).to(self.device)

    def _extract_frames(self, video_path: str) -> List:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        interval = max(1, total // self.max_frames)
        frames = []
        for frame_idx in range(0, total, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if len(frames) >= self.max_frames:
                break
        cap.release()
        return frames

    def generate_description(self, video_path: str) -> str:
        """
        Returns a single string describing the video,
        by captioning up to `max_frames` key frames.
        """
        frames = self._extract_frames(video_path)
        captions = []

        for frame in frames:
            # convert BGR→RGB PIL style
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.pipe:
                out = self.pipe(img, max_new_tokens=50)
                text = out[0].get("generated_text", "").strip()
            else:
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                out_ids = self.model.generate(**inputs, max_new_tokens=50)
                text = self.processor.decode(out_ids[0], skip_special_tokens=True).strip()

            if text:
                captions.append(text)

        # join frame captions into one coherent description
        return " ".join(captions).strip()