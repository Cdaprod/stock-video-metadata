import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from config import (
    VIDEO_EXTENSIONS,
    should_include_directory,
    should_include_video_file,
    transform_batch_name,
    get_output_path,
    get_batch_paths
)

def discover_video_batches() -> dict[str, list[Path]]:
    """
    Discover batches of video files from configured batch root paths.
    Applies ignore rules, size filters, and name transforms.
    """
    batches = defaultdict(list)

    for root in get_batch_paths():
        if not root.exists() or not root.is_dir():
            print(f"❌ Invalid batch path: {root}")
            continue

        print(f"🔍 Searching in: {root}")

        for batch_dir in root.iterdir():
            if not batch_dir.is_dir():
                continue
            if not should_include_directory(batch_dir.name, batch_dir):
                continue

            transformed_name = transform_batch_name(batch_dir.name)

            for file in batch_dir.iterdir():
                if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS:
                    if should_include_video_file(file):
                        batches[transformed_name].append(file)

        if not batches:
            print(f"⚠️  No batches found in: {root}")

    print(f"🎯 Found {len(batches)} batch(es).")
    return dict(batches)

def save_inventory(batches: dict[str, list[Path]], out_json: Path = None, out_csv: Path = None):
    """
    Saves discovered batch info to JSON and CSV.
    """
    records = []

    for batch, files in batches.items():
        for f in files:
            try:
                records.append({
                    'batch_name': batch,
                    'filename': f.name,
                    'full_path': str(f.resolve()),
                    'size_bytes': f.stat().st_size if f.exists() else 0,
                    'size_mb': round(f.stat().st_size / (1024**2), 2) if f.exists() else 0,
                    'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat() if f.exists() else "",
                })
            except Exception as e:
                print(f"⚠️ Error processing file {f}: {e}")

    if not out_json:
        out_json = get_output_path("batch_metadata.json")
    if not out_csv:
        out_csv = get_output_path("video_inventory.csv")

    try:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, 'w') as j:
            json.dump(records, j, indent=2)

        pd.DataFrame(records).to_csv(out_csv, index=False)
        print(f"✅ Inventory saved to:\n  - {out_json}\n  - {out_csv}")
    except Exception as e:
        print(f"❌ Error saving inventory: {e}")