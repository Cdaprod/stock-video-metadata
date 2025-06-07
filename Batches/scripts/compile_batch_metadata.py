/scripts/compile_batch_metadata.py

import os
import pandas as pd

# Path to your SMB-mount. Adjust as needed for Juno, Files app, or local mount.
BASE_DIR = "/Volumes/StockFootage/Batches"
# or for iOS Files App, something like: "/private/var/mobile/Library/Mobile Documents/com~apple~CloudDocs/Batches"

# Metadata CSV file: must have columns like 'filename', 'batch', 'description', 'keywords', etc.
METADATA_CSV = os.path.join(BASE_DIR, "metadata.csv")

def find_videos(base_dir):
    video_exts = (".mp4", ".mov", ".avi", ".mkv")
    records = []
    for batch in os.listdir(base_dir):
        batch_dir = os.path.join(base_dir, batch)
        if not os.path.isdir(batch_dir):
            continue
        for fname in os.listdir(batch_dir):
            if fname.lower().endswith(video_exts):
                records.append({
                    "batch": batch,
                    "filename": fname,
                    "filepath": os.path.join(batch_dir, fname)
                })
    return pd.DataFrame(records)

def main():
    # Load your metadata file (create a blank DataFrame if missing)
    if os.path.exists(METADATA_CSV):
        metadata = pd.read_csv(METADATA_CSV)
    else:
        metadata = pd.DataFrame(columns=["batch", "filename", "description", "keywords"])

    # Find all video files in batches
    files_df = find_videos(BASE_DIR)

    # Merge with metadata on filename
    merged = pd.merge(files_df, metadata, on="filename", how="left", suffixes=("", "_meta"))

    # Show in Jupyter or export as needed
    print(merged.head())
    merged.to_csv(os.path.join(BASE_DIR, "compiled_batch_metadata.csv"), index=False)
    return merged

if __name__ == "__main__":
    main()