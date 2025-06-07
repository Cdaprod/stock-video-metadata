#!/usr/bin/env python3
"""
Video Batch Detection and Metadata Compilation
Handles video files batched by their parent directory names
"""

import os
import json
import glob
from pathlib import Path
from collections import defaultdict
import hashlib
from datetime import datetime

class VideoBatchProcessor:
def **init**(self, root_path="."):
self.root_path = Path(root_path)
self.video_extensions = {’.mp4’, ‘.mov’, ‘.avi’, ‘.mkv’, ‘.m4v’, ‘.wmv’, ‘.flv’, ‘.webm’}

def discover_batches(self):
    """Discover video batches based on parent directory structure"""
    batches = defaultdict(list)
    
    # Scan all subdirectories for video files
    for item in self.root_path.iterdir():
        if item.is_dir():
            batch_name = item.name
            videos = []
            
            # Find video files in this directory
            for video_file in item.iterdir():
                if video_file.is_file() and video_file.suffix.lower() in self.video_extensions:
                    videos.append({
                        'filename': video_file.name,
                        'path': str(video_file.relative_to(self.root_path)),
                        'size_bytes': video_file.stat().st_size,
                        'modified': datetime.fromtimestamp(video_file.stat().st_mtime).isoformat(),
                        'batch': batch_name
                    })
            
            if videos:
                batches[batch_name] = videos
                print(f"📁 Found batch '{batch_name}': {len(videos)} videos")
    
    return dict(batches)

def generate_batch_metadata(self):
    """Generate comprehensive metadata for all video batches"""
    batches = self.discover_batches()
    
    metadata = {
        'scan_time': datetime.now().isoformat(),
        'root_path': str(self.root_path.absolute()),
        'total_batches': len(batches),
        'total_videos': sum(len(videos) for videos in batches.values()),
        'batches': {}
    }
    
    for batch_name, videos in batches.items():
        total_size = sum(video['size_bytes'] for video in videos)
        
        metadata['batches'][batch_name] = {
            'video_count': len(videos),
            'total_size_bytes': total_size,
            'total_size_gb': round(total_size / (1024**3), 2),
            'videos': videos,
            'batch_id': hashlib.md5(batch_name.encode()).hexdigest()[:8]
        }
    
    return metadata

def save_metadata(self, filename="batch_metadata.json"):
    """Save batch metadata to JSON file"""
    metadata = self.generate_batch_metadata()
    
    output_path = self.root_path / filename
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Saved metadata to: {output_path}")
    return metadata

def print_batch_summary(self):
    """Print a summary of discovered batches"""
    batches = self.discover_batches()
    
    if not batches:
        print("❌ No video batches found in current directory")
        print("Expected structure: DirectoryName/video.mp4")
        return
    
    print(f"\n🎬 Video Batch Summary")
    print("=" * 50)
    
    total_videos = 0
    total_size = 0
    
    for batch_name, videos in batches.items():
        batch_size = sum(video['size_bytes'] for video in videos)
        total_videos += len(videos)
        total_size += batch_size
        
        print(f"📁 {batch_name}")
        print(f"   Videos: {len(videos)}")
        print(f"   Size: {batch_size / (1024**3):.2f} GB")
        print()
    
    print(f"📊 Total: {len(batches)} batches, {total_videos} videos, {total_size / (1024**3):.2f} GB")

def main():
processor = VideoBatchProcessor()

# Print summary
processor.print_batch_summary()

# Generate and save metadata
metadata = processor.save_metadata()

# Create workspace-compatible structure info
workspace_info = {
    'workspace_type': 'video_batching',
    'structure': 'parent_directory_batching',
    'batches_discovered': list(metadata['batches'].keys()),
    'ready_for_processing': True
}

with open('workspace_info.json', 'w') as f:
    json.dump(workspace_info, f, indent=2)

print("✅ Workspace analysis complete!")
```

if **name** == "**main**":
main()