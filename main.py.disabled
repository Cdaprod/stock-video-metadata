#!/usr/bin/env python3

"""
main.py
--------
Standalone pipeline runner for batch video metadata enrichment.
Migrates notebook workflow to scriptable, reusable modules.
"""

import os
import sys
from pathlib import Path
import pandas as pd

# === 1. Environment Setup & Imports ===

def setup_environment():
    os.environ["LOCAL_LLM_PATH"] = r"D:/Models/llama2/llama-2-7b-chat.Q4_K_M.gguf"
    os.environ["LOCAL_VTT_PATH"] = r"D:/Models/blip2-opt-2.7b"
    repo_root = Path.cwd()
    scripts_path = repo_root / "scripts"
    if not scripts_path.is_dir():
        raise FileNotFoundError("Missing 'scripts/' directory in repo root")
    sys.path.insert(0, str(scripts_path))
    return repo_root

repo_root = setup_environment()

# Import pipeline modules after sys.path modification
from config import get_smb_root, get_repo_root
from discovery import discover_video_batches, save_inventory
from enrichment import VideoEnricher
from export import export_blackbox_csv, export_blackbox_xml
from upload import upload_batch_or_zip

# === 2. Path Setup ===
batches_root = get_smb_root()
metadata_dir = repo_root / "metadata"
metadata_dir.mkdir(exist_ok=True)

print(f"📂 Repo root:    {repo_root}")
print(f"📂 Batches root: {batches_root}")
print(f"📂 Metadata dir: {metadata_dir}")

# === 3. Load or Discover Inventory ===

def load_or_discover_inventory(batches_root, metadata_dir):
    from pprint import pprint
    batches = {}
    df_inventory = None
    try:
        roots = batches_root if isinstance(batches_root, (list, tuple)) else [batches_root]
        valid = [Path(r) for r in roots if Path(r).exists()]

        if valid:
            root = valid[0]
            print(f"📂 Using discovered video batches from: {root}")
            batches = discover_video_batches()
            if batches:
                save_inventory(
                    batches,
                    out_json = metadata_dir / "batch_metadata.json",
                    out_csv  = metadata_dir / "video_inventory.csv"
                )
                print(f"✅ Discovered {len(batches)} batches and saved metadata.")
            else:
                print("⚠️ No batches discovered.")

        elif (metadata_dir / "video_inventory.csv").exists():
            inv_csv = metadata_dir / "video_inventory.csv"
            print(f"📁 Loading video inventory from CSV: {inv_csv}")
            df_inventory = pd.read_csv(inv_csv)
            print(f"✅ Loaded {len(df_inventory)} entries from CSV.")
            batches = None
        else:
            raise FileNotFoundError("Neither a valid batches_root nor video_inventory.csv was found.")

        # preview
        if batches:
            pprint({k: len(v) for k, v in batches.items()})
        elif df_inventory is not None:
            print(df_inventory[["filename", "full_path"]].head())
        return batches, df_inventory
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Ensure `batches_root` is correct or `metadata/video_inventory.csv` exists.")
        return None, None
    except Exception as e:
        print(f"🚨 Unexpected error:\n{e}")
        return None, None

# === 4. Enrich Inventory ===

def enrich_inventory(metadata_dir):
    enricher = VideoEnricher()
    inv_csv = metadata_dir / "video_inventory.csv"
    out_csv = metadata_dir / "enriched_videos.csv"
    df = pd.read_csv(inv_csv)
    df_enriched = enricher.enrich_dataframe(df, enriched_csv=str(out_csv))
    df_enriched.to_csv(out_csv, index=False)
    print(f"✅ Wrote enriched metadata to: {out_csv}")
    return df_enriched

# === 5. Optional: Use VideoArtifact abstraction (can expand as needed) ===

def process_batches_with_artifacts(batches, metadata_dir):
    from scripts.VideoArtifact import VideoArtifact
    enricher = VideoEnricher()
    for batch_name, files in batches.items():
        artifact = VideoArtifact(batch_name, files, metadata_dir)
        artifact.save_inventory()
        artifact.enrich(enricher)

# === 6. Main Orchestrator ===

def main():
    batches, df_inventory = load_or_discover_inventory(batches_root, metadata_dir)

    # Option 1: Per-batch artifact processing (recommended for next-gen pipeline)
    if batches:
        process_batches_with_artifacts(batches, metadata_dir)
    # Option 2: Simple enrichment of whole inventory
    elif df_inventory is not None:
        enrich_inventory(metadata_dir)
    else:
        print("Nothing to process. Exiting.")

if __name__ == "__main__":
    main()