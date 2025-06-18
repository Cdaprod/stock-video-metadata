# processor.py

from VideoArtifact import ArtifactFactory, BatchProcessor

def process_from_paths(paths, batch_name=None, config=None, interface='cli', interface_metadata=None):
    import argparse
    dummy_args = argparse.Namespace()
    dummy_args.input = paths
    dummy_args.batch_name = batch_name
    dummy_args.quality = (config or {}).get('quality', 'medium')
    dummy_args.format = (config or {}).get('format', 'mp4')
    dummy_args.output = (config or {}).get('output_dir', './outputs')

    batch = ArtifactFactory.create_batch_from_cli(dummy_args, paths)
    processor = BatchProcessor()
    completed_batch = processor.process_batch(batch)
    return completed_batch.get_processing_summary(), completed_batch

def process_from_uploads(uploads, config=None, metadata=None):
    batch = ArtifactFactory.create_batch_from_api(
        uploads=uploads,
        config=config or {},
        request_metadata=metadata or {},
    )
    processor = BatchProcessor()
    completed_batch = processor.process_batch(batch)
    return completed_batch.get_processing_summary(), completed_batch