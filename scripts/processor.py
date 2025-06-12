#!/usr/bin/env python3
"""
Video Batch Detection and Metadata Compilation (robust & notebook-compatible)
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import pandas as pd

class VideoBatchProcessor:
    def __init__(self, root_path=".", verbose=True):
        self.root_path = Path(root_path)
        self.video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.wmv', '.flv', '.webm'}
        self.verbose = verbose

    def discover_batches(self) -> dict[str, list[Path]]:
        batches = defaultdict(list)
        if not self.root_path.exists() or not self.root_path.is_dir():
            if self.verbose:
                print(f"❌ Batch root does not exist or is not a directory: {self.root_path}")
            return {}
        for item in self.root_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                videos = []
                for f in item.iterdir():
                    if f.is_file() and f.suffix.lower() in self.video_extensions and not f.name.startswith('.'):
                        try:
                            videos.append(f)
                        except Exception as e:
                            if self.verbose:
                                print(f"⚠️  Skipping unreadable file: {f} ({e})")
                if videos:
                    batches[item.name] = videos
        if self.verbose:
            print(f"🗂️ Discovered {len(batches)} batches under {self.root_path}")
        return dict(batches)

    def generate_metadata(self) -> dict:
        batches = self.discover_batches()
        meta = {
            'scan_time': datetime.now().isoformat(),
            'root_path': str(self.root_path.resolve()),
            'total_batches': len(batches),
            'total_videos': sum(len(v) for v in batches.values()),
            'batches': {}
        }
        for batch_name, vids in batches.items():
            total_bytes = sum(f.stat().st_size for f in vids if f.exists())
            meta['batches'][batch_name] = {
                'video_count': len(vids),
                'total_size_bytes': total_bytes,
                'total_size_gb': round(total_bytes / (1024**3), 2),
                'videos': [
                    {
                        'batch_name': batch_name,   # match CSV/JSON schema
                        'filename': f.name,
                        'full_path': str(f.resolve()),
                        'size_bytes': f.stat().st_size if f.exists() else 0,
                        'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat() if f.exists() else "",
                    }
                    for f in vids
                ],
                'batch_id': hashlib.md5(batch_name.encode()).hexdigest()[:8]
            }
        return meta
        
    def get_flat_inventory(self) -> pd.DataFrame:
        """Returns a flat DataFrame of all videos across batches with standard metadata fields."""
        meta = self.generate_metadata()
        all_records = []
        for batch in meta.get('batches', {}).values():
            all_records.extend(batch.get('videos', []))
        return pd.DataFrame(all_records)
        
    def save_metadata(self, out_path: Path, as_csv=False):
        meta = self.generate_metadata()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(meta, f, indent=2)
        if self.verbose:
            print(f"💾 Saved metadata to {out_path}")

        # Optional: Write flat CSV inventory
        if as_csv:
            all_records = []
            for batch in meta['batches'].values():
                all_records.extend(batch['videos'])
            df = pd.DataFrame(all_records)
            csv_path = out_path.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            if self.verbose:
                print(f"💾 Saved flat inventory to {csv_path}")
            return meta, df
        return meta

    def print_summary(self):
        batches = self.discover_batches()
        if not batches:
            print("❌ No batches found under", self.root_path)
            return
        print("\n🎬 Video Batch Summary\n" + "="*40)
        total_v = total_bytes = 0
        for name, vids in batches.items():
            sz = sum(f.stat().st_size for f in vids if f.exists())
            print(f"• {name}: {len(vids)} videos, {sz/(1024**3):.2f} GB")
            total_v += len(vids)
            total_bytes += sz
        print(f"\n📊 Total: {len(batches)} batches, {total_v} videos, {total_bytes/(1024**3):.2f} GB\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Discover and save video batch metadata")
    parser.add_argument("--root",       type=str,  default=".", help="Root folder to scan")
    parser.add_argument("--out-json",   type=str,  default="metadata/batch_metadata.json")
    parser.add_argument("--csv",        action="store_true", help="Also export CSV inventory")
    args = parser.parse_args()

    proc = VideoBatchProcessor(root_path=args.root)
    proc.print_summary()
    proc.save_metadata(Path(args.out_json), as_csv=args.csv)

if __name__ == "__main__":
    main()