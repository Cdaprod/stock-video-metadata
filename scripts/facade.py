# /scripts/facade.py
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from components.VideoArtifact import BatchArtifact, VideoArtifact, ArtifactFactory, BatchProcessor
from discovery import discover_video_batches, save_inventory
from enrichment import VideoEnricher
from export import export_blackbox_csv, export_blackbox_xml

# /scripts/facade.py

class VideoPipelineFacade:
    """
    Facade for all entrypoints: CLI, API, Stream, Upload, etc.
    - Handles batch/video ingestion, enrichment, and export.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.enricher = VideoEnricher()  # may accept config in future

    def ingest_from_files(self, paths: List[Union[str, Path]], batch_name: Optional[str] = None) -> BatchArtifact:
        """
        Ingest a batch from local files or directories.
        """
        batch = ArtifactFactory.create_batch_from_cli(
            args=type("FakeArgs", (object,), {
                "batch_name": batch_name or None,
                "quality": "medium",
                "format": "mp4",
                "output": "./outputs"
            })(),
            file_paths=paths
        )
        return batch

    def ingest_from_upload(self, uploads: List[Dict], config: dict) -> BatchArtifact:
        """
        Ingest a batch from direct file uploads (API/iOS shortcut).
        """
        return ArtifactFactory.create_batch_from_api(uploads, config)

    def ingest_from_stream(self, stream_path: str, batch_name: Optional[str] = None) -> BatchArtifact:
        """
        Ingest from a recorded stream (e.g., NDI or RTSP).
        """
        # For a single stream, just wrap in list
        return self.ingest_from_files([stream_path], batch_name)

    def discover_batches(self, root_dirs: List[Union[str, Path]] = None) -> Dict[str, List[Path]]:
        """
        Discover available video batches.
        """
        # Optionally override the batch discovery root
        if root_dirs:
            from config import set_batch_paths
            set_batch_paths(root_dirs)
        return discover_video_batches()

    def enrich_and_export_batch(self, batch: BatchArtifact, export_dir: str, to_csv=True, to_xml=True) -> Dict[str, Any]:
        """
        Run enrichment and export on a batch.
        """
        # Flatten videos to DataFrame
        import pandas as pd
        rows = [v.to_dict() for v in batch.videos]
        df = pd.DataFrame(rows)
        df_enriched = self.enricher.enrich_dataframe(df)
        batch_dir = Path(export_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)
        out_csv = batch_dir / f"{batch.name}_enriched.csv"
        out_xml = batch_dir / f"{batch.name}_metadata.xml"
        if to_csv:
            df_enriched.to_csv(out_csv, index=False)
        if to_xml:
            export_blackbox_xml(df_enriched, batch_dir)
        return {
            "csv": str(out_csv) if to_csv else None,
            "xml": str(out_xml) if to_xml else None,
            "df": df_enriched
        }

    def full_pipeline_from_files(self, paths: List[Union[str, Path]], export_dir: str, batch_name: Optional[str] = None):
        """
        One-call full ingest -> enrich -> export from files.
        """
        batch = self.ingest_from_files(paths, batch_name)
        processor = BatchProcessor()
        batch = processor.process_batch(batch)
        results = self.enrich_and_export_batch(batch, export_dir)
        return results

    def full_pipeline_from_stream(self, stream_path: str, export_dir: str, batch_name: Optional[str] = None):
        """
        Full ingest -> enrich -> export for a recorded stream.
        """
        return self.full_pipeline_from_files([stream_path], export_dir, batch_name)

    # Add more as needed: e.g., ingest_from_ios_shortcut, ingest_from_api, etc.
    
"""
# /main.py or a script

from scripts.facade import VideoPipelineFacade

facade = VideoPipelineFacade()

# From a new batch of files
batch_result = facade.full_pipeline_from_files(
    paths=[
      "B:/Video/OBSBOT Tiny 2/stream1_20240617.mp4",
      "B:/Video/StockFootage/Batches/"
    ],
    export_dir="metadata/",
    batch_name="obsbot_tiny2_20240617"
)
print("Exported:", batch_result)

# Or from a new stream capture
# facade.full_pipeline_from_stream("B:/Video/OBSBOT Tiny 2/capture.mp4", "metadata/")

# For uploads (API/iOS):
# uploads = [{"filename": ..., "data": ..., "metadata": {...}}, ...]
# batch = facade.ingest_from_upload(uploads, config={...})
""" 