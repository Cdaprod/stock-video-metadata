# /scripts/enrichment.py

import cv2
from typing import Optional
from PIL import Image
import pandas as pd
from config import VIDEO_EXTENSIONS

# --- 1) Captioner setup (BLIP/VIT-GPT2) ---
try:
    from transformers import pipeline
    captioner = pipeline(
        "image-to-text",
        model="nlpconnect/vit-gpt2-image-captioning",
        device=-1  # CPU mode
    )
    print("✅ Captioning pipeline loaded")
except Exception as e:
    captioner = None
    print(f"⚠️ Could not load captioning model ({e}); skipping captions")

# --- 2) YOLO setup ---
try:
    from ultralytics import YOLO
    yolom = YOLO("yolov8n.pt")
    has_yolo = True
    print("✅ YOLO detection loaded")
except Exception as e:
    has_yolo = False
    print(f"⚠️ YOLO detection unavailable ({e})")

def generate_caption_and_keywords(img: Image.Image):
    """
    Produce (caption, comma-joined keywords) from a PIL image.
    """
    if captioner is None:
        return "", ""
    try:
        out = captioner(img)[0].get("generated_text", "")
        words = [w.strip(".,").lower() for w in out.split() if len(w) > 3]
        kws = list(dict.fromkeys(words))[:8]
        return out, ", ".join(kws)
    except Exception:
        return "", ""

def extract_middle_frames(path: str, positions: tuple = (0.1, 0.5, 0.9)):
    """Grab a few key frames at given percentage positions of the video. Returns list of raw BGR frames."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total <= 0:
        cap.release()
        return frames
    for pct in positions:
        idx = min(total - 1, int(total * pct))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds these candidate metadata columns:
      - AI_Description:    AI-generated caption
      - AI_Keywords:       AI-generated keywords
      - YOLO_Objects:      YOLO labels
      - Hybrid_Description:AI caption + 'Detected: ...'
      - Filename:          (copy, for BlackBox)
    """
    records = []
    for r in df.itertuples():
        best_cap, best_kw, best_labels = "", "", []
        max_objs = -1

        frames = extract_middle_frames(r.full_path)
        if not frames:
            print(f"⚠️ No frames extracted from {r.filename}")
        for frame in frames:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            try:
                cap_desc, cap_kw = generate_caption_and_keywords(pil)
            except Exception as e:
                print(f"⚠️ Captioning failed for {r.filename}: {e}")
                cap_desc, cap_kw = "", ""

            labels = []
            if has_yolo:
                try:
                    res = yolom.predict(source=pil, imgsz=640, conf=0.3)[0]
                    labels = sorted({yolom.names[int(c)] for c in res.boxes.cls})
                except Exception as e:
                    print(f"⚠️ YOLO failed on {r.filename}: {e}")
                    labels = []

            if len(labels) > max_objs:
                max_objs = len(labels)
                best_cap, best_kw, best_labels = cap_desc, cap_kw, labels

        records.append({
            **r._asdict(),
            "AI_Description": best_cap,
            "AI_Keywords": best_kw,
            "YOLO_Objects": ", ".join(best_labels),
            "Hybrid_Description": (
                f"{best_cap} | Detected: {', '.join(best_labels)}"
                if best_cap and best_labels else best_cap or ""
            ),
            "Filename": r.filename
        })

    cols = [
        "filename", "batch_name", "full_path",
        "AI_Description", "AI_Keywords",
        "YOLO_Objects", "Hybrid_Description", "Filename"
    ]
    return pd.DataFrame(records)[cols]