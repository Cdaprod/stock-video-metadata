import cv2, numpy as np, subprocess, re, json
from pathlib import Path
import pandas as pd
import whisper
from pydub import AudioSegment, silence
from transformers import pipeline
from keybert import KeyBERT
import ffmpeg  # ffmpeg-python

# ---- Model Setup ----
_whisper = whisper.load_model("base")
_FILLERS = {"um", "uh", "like", "you know", "so"}

# HuggingFace + KeyBERT
ner = pipeline("ner", grouped_entities=True)
kw_model = KeyBERT("all-MiniLM-L6-v2")

# ---- Video/Audio Util ----
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
    start = 1
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start), "-i", input_path,
        "-t", str(length_sec), "-an", "-c:v", "copy", output_path
    ], check=True)
    return output_path

def strip_audio(video_path: Path, output_path: Path):
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-c", "copy", "-an", str(output_path)
    ]
    subprocess.run(cmd, check=True)

def trim_video(video_path: Path, output_path: Path, start: float, duration: float):
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ss", str(start), "-t", str(duration),
        "-c", "copy", str(output_path)
    ]
    subprocess.run(cmd, check=True)

def batch_strip_audio(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        input_path = Path(row["full_path"])
        out_path = output_dir / input_path.name
        strip_audio(input_path, out_path)

def batch_trim(df: pd.DataFrame, output_dir: Path, trim_config: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        filename = row["filename"]
        if filename in trim_config:
            start, duration = trim_config[filename]
            input_path = Path(row["full_path"])
            out_path = output_dir / f"trimmed_{filename}"
            trim_video(input_path, out_path, start, duration)

def filter_by_size(df, min_mb=0, max_gb=50):
    return df[(df['size_bytes'] >= min_mb * 1024**2) & (df['size_bytes'] <= max_gb * 1024**3)]

def rename_files(df, pattern_func):
    for i, row in df.iterrows():
        new_name = pattern_func(row['filename'])
        old_path = Path(row['full_path'])
        new_path = old_path.with_name(new_name)
        old_path.rename(new_path)

# ---- Audio/NLP Pipeline ----
def extract_audio(video_path: str, fmt: str = "wav") -> str:
    wav_path = str(Path(video_path).with_suffix(f".{fmt}"))
    (
        ffmpeg
        .input(video_path)
        .output(wav_path, ac=1, ar="16k")
        .overwrite_output()
        .run(quiet=True)
    )
    return wav_path

def transcribe(audio_path: str):
    result = _whisper.transcribe(audio_path, word_timestamps=True)
    words = []
    for seg in result["segments"]:
        for w in seg["words"]:
            words.append((int(w["start"]*1000), int(w["end"]*1000), w["word"]))
    return result["text"], words

def detect_pauses(audio_path: str, min_silence_len_ms: int = 500, silence_thresh_db: int = -40):
    audio = AudioSegment.from_file(audio_path)
    return silence.detect_silence(audio,
                                  min_silence_len=min_silence_len_ms,
                                  silence_thresh=silence_thresh_db)

def detect_fillers(words):
    return [(s,e,w) for s,e,w in words
            if re.sub(r"\W+","",w.lower()) in _FILLERS]

def extract_keywords(text: str, top_n: int = 8):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1,2),
        stop_words='english',
        top_n=top_n
    )
    return [k for k,_ in keywords]

def curate_clip(video_path: str, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = extract_audio(video_path)
    text, words = transcribe(wav)
    pauses  = detect_pauses(wav)
    fillers = detect_fillers(words)
    keywords = extract_keywords(text)
    invalid = any((e-s)>2000 for s,e in pauses) or len(fillers) > 3
    meta = {
        "video": video_path,
        "transcript": text,
        "keywords": keywords,
        "pauses_ms": pauses,
        "fillers": fillers,
        "invalid": invalid
    }
    meta_path = out_dir / f"{Path(video_path).stem}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta