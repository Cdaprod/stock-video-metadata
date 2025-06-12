import cv2
from pathlib import Path
from typing import List, Optional
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

def extract_middle_frames(path: str, positions: tuple = (0.1, 0.5, 0.9)) -> List:
    """
    Grab a few key frames at given percentage positions of the video.
    Returns list of raw BGR frames.
    """
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
    print(f"🎞️ {Path(path).name} -- extracted {len(frames)} frames")
    return frames

def generate_caption_and_keywords(img: Image.Image) -> (str, str):
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
    except Exception as e:
        print(f"⚠️ Captioning error: {e}")
        return "", ""

def detect_yolo_objects(img: Image.Image) -> List[str]:
    """
    Return sorted list of YOLO class names detected in the PIL image.
    """
    if not has_yolo:
        return []
    try:
        results = yolom.predict(source=img, imgsz=640, conf=0.3)[0]
        labels = {yolom.names[int(c)] for c in results.boxes.cls}
        return sorted(labels)
    except Exception as e:
        print(f"⚠️ YOLO detection error: {e}")
        return []

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Idempotently adds/updates these metadata columns on top of `df`:
      - AI_Description        (BLIP caption)
      - AI_Keywords           (from BLIP)
      - YOLO_Objects          (comma-list of detected labels)
      - Hybrid_Description    (caption + "Detected: …")
      - Filename              (mirror of filename for BlackBox)
    Preserves any non-empty existing values.
    """
    # allow lookup of any pre-existing enrichments
    existing = df.set_index("filename", drop=False).to_dict("index")
    records = []

    for r in df.itertuples():
        fname = r.filename
        base = existing.get(fname, {}).copy()

        # start with any prior non-empty values
        best_cap    = base.get("AI_Description", "") or ""
        best_kw     = base.get("AI_Keywords", "")    or ""
        best_labels = (base.get("YOLO_Objects", "").split(", ")
                       if base.get("YOLO_Objects") else [])
        max_objs    = len(best_labels)

        # only override if we find a frame with more objects
        path = r.full_path
        if not Path(path).exists():
            print(f"❌ File not found, skipping: {fname}")
        else:
            for frame in extract_middle_frames(path):
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap_desc, cap_kw = generate_caption_and_keywords(pil)
                labels = detect_yolo_objects(pil)
                if len(labels) > max_objs:
                    max_objs = len(labels)
                    best_cap, best_kw, best_labels = cap_desc, cap_kw, labels

        hybrid = (
            f"{best_cap} | Detected: {', '.join(best_labels)}"
            if best_cap and best_labels else best_cap or ""
        )

        rec = {
            **base,
            "AI_Description":    best_cap,
            "AI_Keywords":       best_kw,
            "YOLO_Objects":      ", ".join(best_labels),
            "Hybrid_Description": hybrid,
            "Filename":          fname
        }
        records.append(rec)

    df_out = pd.DataFrame.from_records(records)

    # enforce column order and presence
    cols = [
        "filename", "batch_name", "full_path",
        "AI_Description", "AI_Keywords",
        "YOLO_Objects", "Hybrid_Description", "Filename"
    ]
    for c in cols:
        if c not in df_out.columns:
            df_out[c] = ""
    return df_out[cols]