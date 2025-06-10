# /scripts/enrichment.py
import cv2
from PIL import Image
import pandas as pd

# Absolute import from scripts/config
from config import VIDEO_EXTENSIONS

# Attempt to load the captioning pipeline; if it fails, set to None
try:
    from transformers import pipeline
    captioner = pipeline(
        "image-to-text",
        model="nlpconnect/vit-gpt2-image-captioning",
        device=-1  # CPU
    )
    print("✅ Captioning pipeline loaded")
except Exception as e:
    captioner = None
    print(f"⚠️ Could not load captioning model ({e}); skipping captions")

def extract_middle_frame(path: str) -> Image.Image | None:
    """Extract the middle frame as a PIL Image, or return None on failure."""
    try:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    except Exception:
        return None

def generate_caption_and_keywords(img: Image.Image):
    """Return (caption, keywords) or empty strings if captioner unavailable or fails."""
    if captioner is None:
        return "", ""
    try:
        out = captioner(img)[0].get('generated_text', "")
        words = [w.strip('.,').lower() for w in out.split() if len(w) > 3]
        kws = list(dict.fromkeys(words))[:8]
        return out, ", ".join(kws)
    except Exception:
        return "", ""

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds three columns to df:
      - Description (AI-generated or blank)
      - Keywords (comma-separated or blank)
      - Filename (copy of original filename for BlackBox)
    """
    descs, kws_list = [], []
    for _, r in df.iterrows():
        img = extract_middle_frame(r['full_path'])
        if img:
            d, k = generate_caption_and_keywords(img)
        else:
            d, k = "", ""
        descs.append(d)
        kws_list.append(k)

    df = df.copy()
    df['Description'] = descs
    df['Keywords']    = kws_list
    df['Filename']    = df['filename']  # BlackBox requires capital "F"

    return df