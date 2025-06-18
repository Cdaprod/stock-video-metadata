# cli.py

import argparse
from processor import process_from_paths

def main():
    parser = argparse.ArgumentParser(description="Artifact-Driven Video Batch Processor")
    parser.add_argument('input', nargs='+', help="Input files or directories")
    parser.add_argument('--batch-name', type=str, help="Custom batch name")
    parser.add_argument('--quality', choices=['low','medium','high'], default='medium')
    parser.add_argument('--format', choices=['mp4','avi','mov'], default='mp4')
    parser.add_argument('--output', type=str, default='./outputs', help="Output dir")
    args = parser.parse_args()

    config = {'quality': args.quality, 'format': args.format, 'output_dir': args.output}
    summary, batch = process_from_paths(args.input, batch_name=args.batch_name, config=config)
    print(f"\n✓ Batch {batch.name} completed ({batch.id})")
    print(f"  Processed: {summary['processed_videos']}/{summary['total_videos']} | Success: {summary['progress_percentage']:.1f}%")
    print(f"  Manifest: {batch.save_manifest(args.output)}")

if __name__ == '__main__':
    main()