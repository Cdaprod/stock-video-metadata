#!/usr/bin/env python3
from scripts.components.VideoArtifact import ArtifactFactory, BatchProcessor
from scripts.enrichment import enrich_batch
from scripts.export import export_blackbox_csv, export_blackbox_xml

def main():
    # CLI or config-based file list
    batch = ArtifactFactory.create_batch_from_cli(None, ["B:/Video/StockFootage/Batches"])
    processor = BatchProcessor()
    batch = processor.process_batch(batch)
    batch = enrich_batch(batch)
    # Export (to whatever format you need)
    export_blackbox_csv(batch)
    export_blackbox_xml(batch)
    print(f"✅ Pipeline complete for batch: {batch.name}")

if __name__ == "__main__":
    main()