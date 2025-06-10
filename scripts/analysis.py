import cv2
from PIL import Image
from transformers import pipeline
import numpy as np

captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

def extract_middle_frame(path: str) -> Image.Image|None:
    cap = cv2.VideoCapture(path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, count // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret: return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def generate_caption_and_keywords(img: Image.Image):
    out = captioner(img)[0]['generated_text']
    words = [w.strip('.,').lower() for w in out.split() if len(w)>3]
    kw = ', '.join(dict.fromkeys(words).keys()[:8])
    return out, kw