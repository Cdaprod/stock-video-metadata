# /scripts/enrichment.py

import cv2
from pathlib import Path
from typing import List, Dict, Any, Callable
from PIL import Image
import pandas as pd
import torch
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
from scenedetect import detect, ContentDetector
import clip
from keybert import KeyBERT
from concurrent.futures import ThreadPoolExecutor

class VideoEnricher:
    def __init__(
        self,
        caption_model: str = "Salesforce/blip2-opt-2.7b",
        blip_fallback: str = "Salesforce/blip-image-captioning-base",
        yolo_model: str = "yolov8n.pt",
        yolo_conf: float = 0.5,
        yolo_iou: float = 0.45,
        clip_model: str = "ViT-B/32",
        clip_threshold: float = 0.3,
    ):
        # device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1) Caption model: try BLIP2, fallback to BLIP
        try:
            self.captioner = pipeline(
                "image-to-text",
                model=caption_model,
                device=0 if self.device=="cuda" else -1,
                use_safetensors=True,
            )
        except Exception:
            self.processor = BlipProcessor.from_pretrained(blip_fallback)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_fallback).to(self.device)
            self.captioner = None

        # 2) YOLO
        self.yolo = YOLO(yolo_model)
        self.yolo_conf, self.yolo_iou = yolo_conf, yolo_iou

        # 3) CLIP
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.device)
        self.clip_threshold = clip_threshold

        # 4) Keyword extractor
        self.kw_model = KeyBERT()

        # register steps
        self.enrichment_steps: List[Callable[[Image.Image, Dict[str,Any]], Dict[str,Any]]] = [
            self._enrich_caption_and_keywords,
            self._enrich_yolo_objects,
            self._enrich_hybrid_description
        ]

    # -- Scene-based frame extraction --
    def extract_scene_frames(self, path: str, max_scenes: int = 3) -> List[Any]:
        # Use modern PySceneDetect API
        scene_list = detect(str(path), ContentDetector())
        
        cap = cv2.VideoCapture(path)
        frames = []
        
        for i, (start, _) in enumerate(scene_list[:max_scenes]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start.get_frames())
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames

    # -- Caption + Keywords with CLIP filtering --
    def _enrich_caption_and_keywords(self, img: Image.Image, ctx: Dict[str,Any]) -> Dict[str,Any]:
        # generate caption
        if self.captioner:
            cap = self.captioner(img, max_new_tokens=30)[0]["generated_text"]
        else:
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            out = self.blip_model.generate(**inputs)
            cap = self.processor.decode(out[0], skip_special_tokens=True)
        # CLIP filter
        image_t = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        text_t  = clip.tokenize([cap]).to(self.device)
        with torch.no_grad():
            score = self.clip_model(image_t, text_t)[0].softmax(dim=-1).item()
        if score < self.clip_threshold:
            cap = ""
        # keywords
        kws = []
        if cap:
            kws = [k for k,_ in self.kw_model.extract_keywords(cap, top_n=8)]
        return {"AI_Description": cap, "AI_Keywords": ", ".join(kws)}

    # -- YOLO with stricter thresholds --
    def _enrich_yolo_objects(self, img: Image.Image, ctx: Dict[str,Any]) -> Dict[str,Any]:
        results = self.yolo.predict(source=img, imgsz=640, conf=self.yolo_conf, iou=self.yolo_iou)[0]
        labels = []
        for c, conf in zip(results.boxes.cls, results.boxes.conf):
            if conf.item() >= self.yolo_conf:
                labels.append(self.yolo.names[int(c)])
        return {"YOLO_Objects": ", ".join(sorted(set(labels)))}

    # -- Hybrid description --
    def _enrich_hybrid_description(self, img: Image.Image, ctx: Dict[str,Any]) -> Dict[str,Any]:
        desc = ctx.get("AI_Description", "")
        objs = ctx.get("YOLO_Objects", "")
        hybrid = f"{desc} | Detected: {objs}" if desc and objs else desc
        return {"Hybrid_Description": hybrid}

    # -- Helpers for DataFrame columns & merging --
    @staticmethod
    def ensure_enrichment_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        return df

    @staticmethod
    def merge_existing_enrichment(df: pd.DataFrame, enriched_csv: str, cols: List[str]) -> pd.DataFrame:
        if Path(enriched_csv).exists():
            old = pd.read_csv(enriched_csv).set_index("filename")
            # Fill NaN values with empty strings in the old data
            old = old.fillna("")
            for c in cols:
                if c in old.columns:
                    df[c] = df.apply(
                        lambda r: old.at[r.filename, c] if not r[c] and r.filename in old.index else r[c],
                        axis=1
                    )
        return df
    
    def _enrich_one(self, row: Any) -> Dict[str, Any]:
        base_raw = row._asdict()

        # Sanitize all values to ensure no NaN or float when expecting str
        base = {
            k: ("" if pd.isna(v) or isinstance(v, float) else v)
            for k, v in base_raw.items()
        }

        path = base["full_path"]
        if not Path(path).exists():
            return {**base, **{k: "" for k in ["AI_Description", "AI_Keywords", "YOLO_Objects", "Hybrid_Description"]}}

        best = {k: base.get(k, "") for k in ["AI_Description", "AI_Keywords", "YOLO_Objects", "Hybrid_Description"]}

        # Count YOLO objects safely
        yolo_val = best.get("YOLO_Objects", "")
        max_objs = len(yolo_val.split(",")) if isinstance(yolo_val, str) and yolo_val.strip() else 0

        frames = self.extract_scene_frames(path) or []
        for f in frames:
            img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            ctx = {**base, **best}
            new = {}
            for step in self.enrichment_steps:
                new.update(step(img, {**ctx, **new}))

            yolo_obj = new.get("YOLO_Objects", "")
            obj_count = len(yolo_obj.split(",")) if isinstance(yolo_obj, str) and yolo_obj.strip() else 0

            if obj_count > max_objs or not best.get("AI_Description"):
                max_objs = obj_count
                best = {**best, **new}

        return {**base, **best, "Filename": base["filename"]}
    
    # -- Main pipeline, with parallelization --
    def enrich_dataframe(self, df: pd.DataFrame, enriched_csv: str = None) -> pd.DataFrame:
        cols = ["AI_Description","AI_Keywords","YOLO_Objects","Hybrid_Description"]
        df = self.ensure_enrichment_columns(df, cols)
        
        # Fill NaN values with empty strings before processing
        df = df.fillna("")
        
        if enriched_csv:
            df = self.merge_existing_enrichment(df, enriched_csv, cols)

        # parallel enrich
        with ThreadPoolExecutor() as exe:
            records = list(exe.map(self._enrich_one, df.itertuples()))

        df_out = pd.DataFrame.from_records(records)
        final_cols = ["filename","batch_name","full_path"] + cols + ["Filename"]
        for c in final_cols:
            if c not in df_out.columns:
                df_out[c] = ""
        return df_out[final_cols]