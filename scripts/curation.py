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