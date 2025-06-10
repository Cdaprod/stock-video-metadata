import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from config import VIDEO_EXTENSIONS

def discover_video_batches(root: Path) -> dict[str, list[Path]]:
    """
    Discover video batches as a mapping of batch_name -> list of Path objects for videos.
    - Skips hidden/system folders and files.
    - Handles missing/misconfigured root gracefully.
    """
    batches = defaultdict(list)
    if not isinstance(root, Path):
        root = Path(root)
    if not root.exists() or not root.is_dir():
        print(f"❌ Provided root does not exist or is not a directory: {root}")
        return {}
    try:
        for p in root.iterdir():
            if p.is_dir() and not p.name.startswith('.'):
                for f in p.iterdir():
                    if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
                        batches[p.name].append(f)
    except Exception as e:
        print(f"🚨 Error during discovery: {e}")
    return dict(batches)

def save_inventory(batches: dict[str, list[Path]], out_json: Path, out_csv: Path):
    """
    Saves discovered batch info to JSON and CSV.
    - Handles non-existent paths, empty batches, IO errors.
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
                    'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat() if f.exists() else "",
                })
            except Exception as e:
                print(f"⚠️  Error processing file {f}: {e}")
    try:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, 'w') as j:
            json.dump(records, j, indent=2)
        pd.DataFrame(records).to_csv(out_csv, index=False)
        print(f"✅ Inventory written to {out_json} and {out_csv}")
    except Exception as e:
        print(f"❌ Error saving inventory: {e}")