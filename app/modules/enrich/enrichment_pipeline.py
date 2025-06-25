import os
import cv2

try:
    import openai
    from openai._base_client import SyncHttpxClientWrapper
    # Only patch if openai is older than 1.56
    if tuple(map(int, openai.__version__.split("."))) <= (1, 55, 3):
        class NoProxiesWrapper(SyncHttpxClientWrapper):
            def __init__(self, *args, **kwargs):
                kwargs.pop("proxies", None)
                super().__init__(*args, **kwargs)
        openai._base_client.SyncHttpxClientWrapper = NoProxiesWrapper
except Exception as e:
    print(f"OpenAI patch failed (may be unnecessary): {e}")
    
import torch
import clip
import json
import pandas as pd

from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
from typing import List, Dict, Any, Callable
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
from scenedetect import detect, ContentDetector
from keybert import KeyBERT
from pytesseract import image_to_string

from .llm import MetadataLLM
from .vtt import VideoToText
from .utils import patch_openai_no_proxy, OPENAI_API_KEY

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

patch_openai_no_proxy()
openai.api_key = OPENAI_API_KEY or openai.api_key

class VideoEnricher:
    def __init__(
        self,
        llm=None,
        caption_model: str = "Salesforce/blip2-opt-2.7b",
        blip_fallback: str = "Salesforce/blip-image-captioning-base",
        yolo_model: str = "yolov8n.pt",
        yolo_conf: float = 0.5,
        yolo_iou: float = 0.45,
        clip_model: str = "ViT-B/32",
        clip_threshold: float = 0.3,
    ):
        self.vtt = VideoToText()
        self.llm = llm if llm is not None else MetadataLLM()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        openai.api_key = os.getenv("OPENAI_API_KEY") or openai.api_key
        if not openai.api_key:
            print("⚠️ OPENAI_API_KEY not set; metadata steps will be skipped", flush=True)

        self._frame_cache: Dict[str, List[Any]] = {}

        try:
            self.captioner = pipeline(
                "image-to-text", model=caption_model,
                device=0 if self.device == "cuda" else -1, use_safetensors=True
            )
        except Exception:
            self.processor = BlipProcessor.from_pretrained(blip_fallback)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_fallback).to(self.device)
            self.captioner = None

        self.yolo = YOLO(yolo_model)
        self.yolo_conf, self.yolo_iou = yolo_conf, yolo_iou
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.device)
        self.clip_threshold = clip_threshold
        self.kw_model = KeyBERT()

        self.steps: List[Callable[[str, Image.Image, Dict[str, Any]], Dict[str, Any]]] = []
        self.register_step(self._enrich_video_caption)
        self.register_step(self._enrich_caption_and_keywords)
        self.register_step(self._enrich_clip_filter)
        self.register_step(self._enrich_yolo_objects)
        self.register_step(self._enrich_hybrid_description)
        self.register_step(self._enrich_ocr)
        self.register_step(self._finalize_metadata)

    def register_step(self, step: Callable) -> None:
        self.steps.append(step)

    def unregister_step(self, step: Callable) -> None:
        self.steps = [s for s in self.steps if s != step]

    def extract_scene_frames(self, path: str, max_scenes: int = 3) -> List[Any]:
        if path in self._frame_cache:
            return self._frame_cache[path]
        scene_list = detect(str(path), ContentDetector())
        cap = cv2.VideoCapture(path)
        frames = []
        for start, _ in scene_list[:max_scenes]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start.get_frames())
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        self._frame_cache[path] = frames
        return frames

    def _enrich_caption_and_keywords(self, path: str, img: Image.Image, ctx: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if self.captioner:
                result = self.captioner(img, max_new_tokens=30)
                cap = result[0].get("generated_text", "") if result else ""
            else:
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                out = self.blip_model.generate(**inputs)
                cap = self.processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            print(f"⚠️ Caption generation failed: {e}", flush=True)
            cap = ""

        kws = []
        if cap:
            try:
                kws = [k for k, _ in self.kw_model.extract_keywords(cap, top_n=8)]
            except Exception as e:
                print(f"⚠️ Keyword extraction failed: {e}", flush=True)

        return {
            "AI_Description": cap,
            "AI_Keywords": ", ".join(kws)
        }

    def _enrich_clip_filter(self, path: str, img: Image.Image, ctx: Dict[str, Any]) -> Dict[str, Any]:
        text = ctx.get("AI_Description", "")
        if not text:
            return {}
        image_t = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        text_t = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            score = self.clip_model(image_t, text_t)[0].softmax(dim=-1).item()
        return {"CLIP_Score": score, "CLIP_Pass": score >= self.clip_threshold}

    def _enrich_yolo_objects(self, path: str, img: Image.Image, ctx: Dict[str, Any]) -> Dict[str, Any]:
        results = self.yolo.predict(source=img, imgsz=640, conf=self.yolo_conf, iou=self.yolo_iou)[0]
        labels = []
        for c, conf in zip(results.boxes.cls, results.boxes.conf):
            if conf.item() >= self.yolo_conf:
                labels.append(self.yolo.names[int(c)])
        return {"YOLO_Objects": ", ".join(sorted(set(labels)))}

    def _enrich_hybrid_description(self, path: str, img: Image.Image, ctx: Dict[str, Any]) -> Dict[str, Any]:
        desc = ctx.get("AI_Description", "")
        objs = ctx.get("YOLO_Objects", "")
        hybrid = f"{desc} | Detected: {objs}" if desc and objs else desc
        return {"Hybrid_Description": hybrid}

    def _enrich_ocr(self, path: str, img: Image.Image, ctx: Dict[str, Any]) -> Dict[str, Any]:
        try:
            text = image_to_string(img)
        except Exception:
            text = ""
        return {"OCR_Text": text}

    def _extract_metadata(self, caption: str, objects: str) -> Dict[str, Any]:
        if not openai.api_key:
            return {"SceneType": "", "SceneObjects": "", "MainAction": "", "SceneMood": ""}
        prompt = f"Describe the type of scene, key objects, main action, and mood for: '{caption}' and detected objects: '{objects}'."
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"⚠️ Metadata extraction failed: {e}", flush=True)
            return {"SceneType": "", "SceneObjects": "", "MainAction": "", "SceneMood": ""}

    def _finalize_metadata(self, path: str, img: Image.Image, ctx: Dict[str, Any]) -> Dict[str, Any]:
        if "meta_done" in ctx:
            return {}
        meta = self._extract_metadata(ctx.get("AI_Description", ""), ctx.get("YOLO_Objects", ""))
        return {
            "SceneType": meta["SceneType"],
            "SceneObjects": meta["SceneObjects"],
            "MainAction": meta["MainAction"],
            "SceneMood": meta["SceneMood"],
            "meta_done": True
        }

    def _refine_metadata(self, initial_meta):
        return self.llm.refine_metadata(initial_meta)
        
    def _enrich_video_caption(self, path: str, img: Image.Image, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Try a full-video caption first. If it fails or is empty, 
        the next step (_enrich_caption_and_keywords) will run as a fallback.
        """
        try:
            desc = self.vtt.generate_description(path)
            if desc:
                return {"AI_Description": desc}
        except Exception as e:
            print(f"[vtt.py] ⚠️ VideoToText failed: {e}", flush=True)
        return {}

    def _enrich_one(self, row: Any) -> Dict[str, Any]:
        base = row._asdict()
        path = base["full_path"]
        best = {
            "AI_Description": "",
            "AI_Keywords": "",
            "CLIP_Score": "",
            "CLIP_Pass": "",
            "YOLO_Objects": "",
            "Hybrid_Description": "",
            "OCR_Text": "",
            "SceneType": "",
            "SceneObjects": "",
            "MainAction": "",
            "SceneMood": "",
        }
        # best = {}
        frames = self.extract_scene_frames(path)
        for f in frames:
            img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            ctx = {**base, **best}
            for step in self.steps:
                res = step(path, img, ctx)
                ctx.update(res)
            best = {**best, **ctx}
        
        # Without RAG
        # return {
        #     **base,
        #     **{k: best.get(k, "") for k in [
        #         "AI_Description", "AI_Keywords", "CLIP_Score", "CLIP_Pass",
        #         "YOLO_Objects", "Hybrid_Description", "OCR_Text",
        #         "SceneType", "SceneObjects", "MainAction", "SceneMood"
        #     ]},
        #     "Filename": base.get("filename", "")
        # }

        # Final RAG refinement step
        refined_metadata = self._refine_metadata(best)
        
        return {
            **base,
            **best,
            **refined_metadata,
            "Filename": base["filename"]
        }
         
    def enrich_dataframe(self, df: pd.DataFrame, enriched_csv: str = None) -> pd.DataFrame:
        cols = [
            "AI_Description", "AI_Keywords", "CLIP_Score", "CLIP_Pass",
            "YOLO_Objects", "Hybrid_Description", "OCR_Text",
            "SceneType", "SceneObjects", "MainAction", "SceneMood"
        ]
        self.ensure_enrichment_columns(df, cols)
        records = []
        for row in tqdm(df.itertuples(), desc="Enriching videos"):
            records.append(self._enrich_one(row))
        return pd.DataFrame.from_records(records)[["filename", "batch_name", "full_path"] + cols]

    @staticmethod
    def ensure_enrichment_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        return df