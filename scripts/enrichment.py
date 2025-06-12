import cv2
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from PIL import Image
import pandas as pd
from config import VIDEO_EXTENSIONS
import torch, torchvision

print(torch.__version__)
print(torchvision.__version__)

class VideoEnricher:
    def __init__(self, caption_model="nlpconnect/vit-gpt2-image-captioning", yolo_model="yolov8n.pt"):
        self.captioner = None
        self.yolom = None
        self.has_yolo = False
        self.enrichment_steps: List[Callable[[Image.Image, Dict[str, Any]], Dict[str, Any]]] = []
        self._register_default_enrichments()

        # Load BLIP caption model
        try:
            from transformers import pipeline
            self.captioner = pipeline(
                "image-to-text",
                model=caption_model,
                device=-1  # CPU
            )
            print("✅ Captioning pipeline loaded")
        except Exception as e:
            print(f"⚠️ Could not load captioning model: {e}")

        # Load YOLO model
        try:
            from ultralytics import YOLO
            self.yolom = YOLO(yolo_model)
            self.has_yolo = True
            print("✅ YOLO detection loaded")
        except Exception as e:
            print(f"⚠️ YOLO detection unavailable: {e}")

    def _register_default_enrichments(self):
        self.enrichment_steps = [
            self._enrich_caption_and_keywords,
            self._enrich_yolo_objects,
            self._enrich_hybrid_description
        ]

    def add_enrichment_step(self, step_fn: Callable[[Image.Image, Dict[str, Any]], Dict[str, Any]]):
        self.enrichment_steps.append(step_fn)

    def validate_paths(self, df, path_col="full_path", max_print=5):
        """Check if all files exist."""
        missing = 0
        for _, row in df.iterrows():
            exists = Path(row[path_col]).exists()
            if not exists:
                missing += 1
                if missing <= max_print:
                    print(f"❌ File missing: {row.get('filename','?')} -> {row[path_col]}")
        if missing:
            print(f"⚠️ {missing} files missing in inventory.")
        else:
            print("✅ All inventory files found.")

    @staticmethod
    def ensure_enrichment_columns(df, enrichment_cols):
        for col in enrichment_cols:
            if col not in df.columns:
                df[col] = ""
        return df

    @staticmethod
    def merge_existing_enrichment(df, enriched_csv, enrichment_cols):
        if Path(enriched_csv).exists():
            df_enriched = pd.read_csv(enriched_csv)
            print(f"🧠 Merging from prior enrichment: {len(df_enriched)} rows")
            df_enriched = df_enriched.set_index("filename")
            for col in enrichment_cols:
                if col in df_enriched.columns:
                    df[col] = df.apply(
                        lambda row: df_enriched.at[row['filename'], col]
                            if ((col not in row) or not str(row[col]).strip())
                            and row['filename'] in df_enriched.index
                            and pd.notna(df_enriched.at[row['filename'], col])
                            else row[col],
                        axis=1
                    )
        return df

    def extract_middle_frames(self, path: str, positions: tuple = (0.1, 0.5, 0.9)) -> List:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            print(f"❌ Invalid video or zero frames: {path}")
            cap.release()
            return []
        frames = []
        for pct in positions:
            idx = min(total - 1, int(total * pct))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ Failed to read frame {idx} from {path}")
            else:
                frames.append(frame)
        cap.release()
        return frames

    def _enrich_caption_and_keywords(self, pil_img: Image.Image, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.captioner is None:
            return {"AI_Description": "", "AI_Keywords": ""}
        try:
            out = self.captioner(pil_img, max_new_tokens=30)[0].get("generated_text", "")
            words = [w.strip(".,").lower() for w in out.split() if len(w) > 3]
            kws = list(dict.fromkeys(words))[:8]
            return {"AI_Description": out, "AI_Keywords": ", ".join(kws)}
        except Exception as e:
            print(f"⚠️ Captioning error: {e}")
            return {"AI_Description": "", "AI_Keywords": ""}

    def _enrich_yolo_objects(self, pil_img: Image.Image, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.has_yolo:
            return {"YOLO_Objects": ""}
        try:
            results = self.yolom.predict(source=pil_img, imgsz=640, conf=0.3)[0]
            labels = {self.yolom.names[int(c)] for c in results.boxes.cls}
            return {"YOLO_Objects": ", ".join(sorted(labels))}
        except Exception as e:
            print(f"⚠️ YOLO detection error: {e}")
            return {"YOLO_Objects": ""}

    def _enrich_hybrid_description(self, pil_img: Image.Image, context: Dict[str, Any]) -> Dict[str, Any]:
        desc = context.get("AI_Description", "")
        yolo_objs = context.get("YOLO_Objects", "")
        hybrid = (f"{desc} | Detected: {yolo_objs}" if desc and yolo_objs else desc or "")
        return {"Hybrid_Description": hybrid}

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        enrichment_cols = [
            "AI_Description", "AI_Keywords", "YOLO_Objects", "Hybrid_Description"
        ]
        df = self.ensure_enrichment_columns(df, enrichment_cols)
        records = []
        for r in df.itertuples():
            fname = r.filename
            base = r._asdict()
            path = r.full_path
            if not Path(path).exists():
                print(f"❌ File not found, skipping: {fname}")
                rec = {**base, **{col: "" for col in enrichment_cols}}
                records.append(rec)
                continue

            best_result = {col: base.get(col, "") for col in enrichment_cols}
            max_objs = len(best_result.get("YOLO_Objects", "").split(", ")) if best_result.get("YOLO_Objects") else 0

            for frame in self.extract_middle_frames(path):
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                enrichments = {}
                context = {**base, **best_result}
                for step in self.enrichment_steps:
                    enrichments.update(step(pil_img, context | enrichments))
                obj_count = len(enrichments.get("YOLO_Objects", "").split(", ")) if enrichments.get("YOLO_Objects") else 0
                if obj_count > max_objs or not best_result["AI_Description"]:
                    max_objs = obj_count
                    best_result = {**best_result, **enrichments}

            rec = {**base, **best_result, "Filename": fname}
            records.append(rec)

        cols = [
            "filename", "batch_name", "full_path",
            "AI_Description", "AI_Keywords",
            "YOLO_Objects", "Hybrid_Description", "Filename"
        ]
        df_out = pd.DataFrame.from_records(records)
        for c in cols:
            if c not in df_out.columns:
                df_out[c] = ""
        return df_out[cols]