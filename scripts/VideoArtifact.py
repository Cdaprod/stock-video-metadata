#!/usr/bin/env python3
"""
Video Batch Processing System using Artifact Pattern

Videos and Batches are treated as first-class artifacts with:
- Immutable state tracking
- Event sourcing capabilities
- Clear responsibility boundaries
- Lifecycle management
"""

import os
import json
import uuid
import hashlib
from datetime import datetime
from typing import List, Union, Dict, Any, Optional, Protocol
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

class ArtifactState(Enum):
    """States that artifacts can be in"""
    CREATED = "created"
    VALIDATED = "validated" 
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"

class ArtifactEvent:
    """Represents an event in an artifact's lifecycle"""
    def __init__(self, event_type: str, data: Dict[str, Any] = None, timestamp: datetime = None):
        self.id = str(uuid.uuid4())
        self.event_type = event_type
        self.data = data or {}
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'event_type': self.event_type,
            'data': self.data,
            'timestamp': self.timestamp.isoformat()
        }

class Artifact(ABC):
    """Base artifact class with state management and event sourcing"""
    
    def __init__(self, artifact_id: str = None):
        self.id = artifact_id or str(uuid.uuid4())
        self.created_at = datetime.now()
        self.state = ArtifactState.CREATED
        self.events: List[ArtifactEvent] = []
        self.metadata: Dict[str, Any] = {}
        self._version = 1
    
    def emit_event(self, event_type: str, data: Dict[str, Any] = None):
        """Emit an event and update artifact state"""
        event = ArtifactEvent(event_type, data)
        self.events.append(event)
        self._handle_event(event)
        self._version += 1
    
    @abstractmethod
    def _handle_event(self, event: ArtifactEvent):
        """Handle state changes based on events"""
        pass
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get complete event history"""
        return [event.to_dict() for event in self.events]
    
    def get_state_at(self, timestamp: datetime) -> Dict[str, Any]:
        """Reconstruct artifact state at a specific point in time"""
        # Implementation would replay events up to timestamp
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize artifact to dictionary"""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate artifact integrity"""
        pass

class VideoArtifact(Artifact):
    """Video as a first-class artifact with lifecycle management"""
    
    def __init__(self, filename: str, source_type: str, artifact_id: str = None):
        super().__init__(artifact_id)
        self.filename = filename
        self.source_type = source_type  # 'file', 'upload', 'stream', 'url'
        self.file_path: Optional[str] = None
        self.file_data: Optional[bytes] = None
        self.file_hash: Optional[str] = None
        self.processing_results: Dict[str, Any] = {}
        self.dependencies: List[str] = []  # IDs of other artifacts this depends on
        
        # Video-specific metadata
        self.duration: Optional[float] = None
        self.resolution: Optional[tuple] = None
        self.codec: Optional[str] = None
        self.bitrate: Optional[int] = None
        self.frame_rate: Optional[float] = None
        
        self.emit_event('video_created', {
            'filename': filename,
            'source_type': source_type
        })
    
    def set_source_data(self, file_path: str = None, file_data: bytes = None):
        """Set the source data for this video artifact"""
        if file_path:
            self.file_path = file_path
            self.file_hash = self._compute_file_hash(file_path)
            self.emit_event('source_attached', {'file_path': file_path, 'hash': self.file_hash})
        elif file_data:
            self.file_data = file_data
            self.file_hash = self._compute_data_hash(file_data)
            self.emit_event('data_attached', {'data_size': len(file_data), 'hash': self.file_hash})
    
    def extract_metadata(self):
        """Extract video metadata (would use ffprobe or similar)"""
        # Placeholder - would use actual video analysis
        if self.file_path and os.path.exists(self.file_path):
            stat = os.stat(self.file_path)
            self.metadata.update({
                'file_size': stat.st_size,
                'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        # Mock metadata extraction
        self.duration = 120.5  # seconds
        self.resolution = (1920, 1080)
        self.codec = 'h264'
        self.bitrate = 5000000  # bps
        self.frame_rate = 30.0
        
        self.emit_event('metadata_extracted', {
            'duration': self.duration,
            'resolution': self.resolution,
            'codec': self.codec
        })
    
    def validate(self) -> bool:
        """Validate video artifact integrity"""
        if not self.filename:
            return False
        
        if self.file_path and not os.path.exists(self.file_path):
            return False
        
        if self.file_data and len(self.file_data) == 0:
            return False
        
        # Validate hash if available
        if self.file_hash:
            if self.file_path:
                current_hash = self._compute_file_hash(self.file_path)
                if current_hash != self.file_hash:
                    self.emit_event('integrity_violation', {'expected': self.file_hash, 'actual': current_hash})
                    return False
        
        self.state = ArtifactState.VALIDATED
        self.emit_event('validated', {'hash': self.file_hash})
        return True
    
    def start_processing(self, processor_config: Dict[str, Any]):
        """Begin processing this video"""
        self.state = ArtifactState.PROCESSING
        self.emit_event('processing_started', processor_config)
    
    def complete_processing(self, results: Dict[str, Any]):
        """Mark processing as complete with results"""
        self.processing_results.update(results)
        self.state = ArtifactState.COMPLETED
        self.emit_event('processing_completed', results)
    
    def fail_processing(self, error: str):
        """Mark processing as failed"""
        self.state = ArtifactState.FAILED
        self.emit_event('processing_failed', {'error': error})
    
    def _handle_event(self, event: ArtifactEvent):
        """Handle video-specific events"""
        if event.event_type == 'processing_started':
            self.state = ArtifactState.PROCESSING
        elif event.event_type == 'processing_completed':
            self.state = ArtifactState.COMPLETED
        elif event.event_type == 'processing_failed':
            self.state = ArtifactState.FAILED
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _compute_data_hash(self, data: bytes) -> str:
        """Compute SHA256 hash of data"""
        return hashlib.sha256(data).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'filename': self.filename,
            'source_type': self.source_type,
            'state': self.state.value,
            'file_path': self.file_path,
            'file_hash': self.file_hash,
            'duration': self.duration,
            'resolution': self.resolution,
            'codec': self.codec,
            'bitrate': self.bitrate,
            'frame_rate': self.frame_rate,
            'metadata': self.metadata,
            'processing_results': self.processing_results,
            'dependencies': self.dependencies,
            'created_at': self.created_at.isoformat(),
            'version': self._version,
            'events': self.get_history()
        }

class BatchArtifact(Artifact):
    """Batch as a coordinating artifact that manages video artifacts"""
    
    def __init__(self, batch_name: str = None, artifact_id: str = None):
        super().__init__(artifact_id)
        self.name = batch_name or f"batch_{self.id[:8]}"
        self.videos: List[VideoArtifact] = []
        self.config: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self.source_interface: Optional[str] = None  # 'cli', 'api', 'ios_shortcuts'
        
        # Batch processing metrics
        self.total_videos = 0
        self.processed_videos = 0
        self.failed_videos = 0
        self.processing_time: Optional[float] = None
        
        self.emit_event('batch_created', {'name': self.name})
    
    def add_video(self, video: VideoArtifact):
        """Add a video artifact to this batch"""
        self.videos.append(video)
        self.total_videos += 1
        video.dependencies.append(self.id)  # Video depends on batch
        
        self.emit_event('video_added', {
            'video_id': video.id,
            'filename': video.filename,
            'total_videos': self.total_videos
        })
    
    def set_configuration(self, config: Dict[str, Any]):
        """Set batch processing configuration"""
        self.config.update(config)
        self.emit_event('configuration_set', config)
    
    def set_source_interface(self, interface: str, interface_data: Dict[str, Any] = None):
        """Record which interface created this batch"""
        self.source_interface = interface
        self.metadata['source_interface'] = interface
        if interface_data:
            self.metadata['interface_data'] = interface_data
        
        self.emit_event('source_interface_set', {
            'interface': interface,
            'data': interface_data
        })
    
    def validate(self) -> bool:
        """Validate batch and all constituent videos"""
        if not self.videos:
            self.emit_event('validation_failed', {'reason': 'no_videos'})
            return False
        
        # Validate all videos
        invalid_videos = []
        for video in self.videos:
            if not video.validate():
                invalid_videos.append(video.id)
        
        if invalid_videos:
            self.emit_event('validation_failed', {
                'reason': 'invalid_videos',
                'invalid_video_ids': invalid_videos
            })
            return False
        
        self.state = ArtifactState.VALIDATED
        self.emit_event('batch_validated', {'video_count': len(self.videos)})
        return True
    
    def start_processing(self):
        """Begin processing all videos in the batch"""
        if not self.validate():
            return False
        
        self.state = ArtifactState.PROCESSING
        self.emit_event('batch_processing_started', {
            'video_count': len(self.videos),
            'config': self.config
        })
        
        # Start processing each video
        for video in self.videos:
            video.start_processing(self.config)
        
        return True
    
    def update_video_progress(self, video_id: str, results: Dict[str, Any] = None, error: str = None):
        """Update progress when a video completes or fails"""
        video = next((v for v in self.videos if v.id == video_id), None)
        if not video:
            return
        
        if error:
            video.fail_processing(error)
            self.failed_videos += 1
            self.emit_event('video_failed', {'video_id': video_id, 'error': error})
        else:
            video.complete_processing(results or {})
            self.processed_videos += 1
            self.emit_event('video_completed', {'video_id': video_id, 'results': results})
        
        # Check if batch is complete
        if self.processed_videos + self.failed_videos == self.total_videos:
            self._complete_batch()
    
    def _complete_batch(self):
        """Mark batch as complete and generate final results"""
        self.state = ArtifactState.COMPLETED
        
        # Aggregate results from all videos
        self.results = {
            'total_videos': self.total_videos,
            'processed_videos': self.processed_videos,
            'failed_videos': self.failed_videos,
            'success_rate': self.processed_videos / self.total_videos if self.total_videos > 0 else 0,
            'video_results': {video.id: video.processing_results for video in self.videos}
        }
        
        self.emit_event('batch_completed', self.results)
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get current processing summary"""
        return {
            'batch_id': self.id,
            'name': self.name,
            'state': self.state.value,
            'total_videos': self.total_videos,
            'processed_videos': self.processed_videos,
            'failed_videos': self.failed_videos,
            'pending_videos': self.total_videos - self.processed_videos - self.failed_videos,
            'progress_percentage': ((self.processed_videos + self.failed_videos) / self.total_videos * 100) if self.total_videos > 0 else 0
        }
    
    def save_manifest(self, output_dir: str = "./batches") -> str:
        """Save complete batch artifact to disk"""
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, f"{self.id}_manifest.json")
        
        with open(manifest_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        self.emit_event('manifest_saved', {'path': manifest_path})
        return manifest_path
    
    def _handle_event(self, event: ArtifactEvent):
        """Handle batch-specific events"""
        if event.event_type == 'batch_processing_started':
            self.state = ArtifactState.PROCESSING
        elif event.event_type == 'batch_completed':
            self.state = ArtifactState.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'state': self.state.value,
            'source_interface': self.source_interface,
            'config': self.config,
            'results': self.results,
            'total_videos': self.total_videos,
            'processed_videos': self.processed_videos,
            'failed_videos': self.failed_videos,
            'processing_time': self.processing_time,
            'videos': [video.to_dict() for video in self.videos],
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'version': self._version,
            'events': self.get_history()
        }

class ArtifactFactory:
    """Factory for creating video and batch artifacts from different sources"""
    
    @staticmethod
    def create_video_from_path(file_path: str) -> VideoArtifact:
        """Create video artifact from file path"""
        filename = Path(file_path).name
        video = VideoArtifact(filename, 'file')
        video.set_source_data(file_path=file_path)
        video.extract_metadata()
        return video
    
    @staticmethod
    def create_video_from_upload(filename: str, data: bytes, metadata: Dict[str, Any] = None) -> VideoArtifact:
        """Create video artifact from uploaded data"""
        video = VideoArtifact(filename, 'upload')
        video.set_source_data(file_data=data)
        if metadata:
            video.metadata.update(metadata)
        video.extract_metadata()
        return video
    
    @staticmethod
    def create_batch_from_cli(args, file_paths: List[str]) -> BatchArtifact:
        """Create batch artifact from CLI arguments"""
        batch_name = getattr(args, 'batch_name', None) or f"cli_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch = BatchArtifact(batch_name)
        
        # Set source interface
        batch.set_source_interface('cli', {
            'args': vars(args),
            'file_paths': file_paths
        })
        
        # Create video artifacts
        for file_path in file_paths:
            if os.path.exists(file_path):
                if os.path.isfile(file_path):
                    video = ArtifactFactory.create_video_from_path(file_path)
                    batch.add_video(video)
                elif os.path.isdir(file_path):
                    # Handle directory
                    for video_file in Path(file_path).rglob('*'):
                        if video_file.is_file() and ArtifactFactory._is_video_file(str(video_file)):
                            video = ArtifactFactory.create_video_from_path(str(video_file))
                            batch.add_video(video)
        
        # Set configuration from args
        config = {
            'quality': getattr(args, 'quality', 'medium'),
            'format': getattr(args, 'format', 'mp4'),
            'output_dir': getattr(args, 'output', './outputs')
        }
        batch.set_configuration(config)
        
        return batch
    
    @staticmethod
    def create_batch_from_api(uploads: List[Dict], config: Dict[str, Any], request_metadata: Dict[str, Any] = None) -> BatchArtifact:
        """Create batch artifact from API upload"""
        batch = BatchArtifact(f"api_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Set source interface
        batch.set_source_interface('api', request_metadata or {})
        
        # Create video artifacts from uploads
        for upload in uploads:
            video = ArtifactFactory.create_video_from_upload(
                filename=upload.get('filename', 'upload.mp4'),
                data=upload.get('data', b''),
                metadata=upload.get('metadata', {})
            )
            batch.add_video(video)
        
        batch.set_configuration(config)
        return batch
    
    @staticmethod
    def create_batch_from_ios(shortcuts_data: Dict[str, Any]) -> BatchArtifact:
        """Create batch artifact from iOS Shortcuts"""
        batch = BatchArtifact(f"ios_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Set source interface
        batch.set_source_interface('ios_shortcuts', {
            'shortcut_name': shortcuts_data.get('shortcut_name'),
            'device_info': shortcuts_data.get('device_info', {})
        })
        
        # Create video artifacts
        for video_data in shortcuts_data.get('videos', []):
            # Handle base64 encoded data from iOS
            import base64
            video_bytes = base64.b64decode(video_data.get('data', ''))
            
            video = ArtifactFactory.create_video_from_upload(
                filename=video_data.get('filename', 'ios_video.mov'),
                data=video_bytes,
                metadata=video_data.get('metadata', {})
            )
            batch.add_video(video)
        
        batch.set_configuration(shortcuts_data.get('config', {}))
        return batch
    
    @staticmethod
    def _is_video_file(path: str) -> bool:
        """Check if file is a video"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        return Path(path).suffix.lower() in video_extensions

class BatchProcessor:
    """Processes batch artifacts through their lifecycle"""
    
    def __init__(self):
        self.active_batches: Dict[str, BatchArtifact] = {}
    
    def process_batch(self, batch: BatchArtifact) -> BatchArtifact:
        """Process a batch artifact through its complete lifecycle"""
        self.active_batches[batch.id] = batch
        
        # Start processing
        if batch.start_processing():
            # Simulate processing each video
            for video in batch.videos:
                try:
                    # This is where your actual inference would happen
                    results = self._process_video(video, batch.config)
                    batch.update_video_progress(video.id, results)
                except Exception as e:
                    batch.update_video_progress(video.id, error=str(e))
        
        # Save final manifest
        batch.save_manifest()
        
        return batch
    
    def _process_video(self, video: VideoArtifact, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single video artifact (mock implementation)"""
        # Mock processing results
        return {
            'output_path': f"./outputs/{video.id}_processed.{config.get('format', 'mp4')}",
            'processing_time': 2.5,
            'inference_results': {
                'confidence': 0.95,
                'predictions': ['object1', 'object2'],
                'frames_processed': int(video.duration * video.frame_rate) if video.duration and video.frame_rate else 0
            }
        }
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active batch"""
        batch = self.active_batches.get(batch_id)
        return batch.get_processing_summary() if batch else None

# Example usage and interfaces
def main():
    """Example CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Batch Processor with Artifact Pattern')
    parser.add_argument('input', nargs='+', help='Input video files or directories')
    parser.add_argument('--batch-name', type=str, help='Custom batch name')
    parser.add_argument('--quality', choices=['low', 'medium', 'high'], default='medium')
    parser.add_argument('--format', choices=['mp4', 'avi', 'mov'], default='mp4')
    parser.add_argument('--output', type=str, default='./outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # Create batch artifact from CLI
    batch = ArtifactFactory.create_batch_from_cli(args, args.input)
    
    # Process the batch
    processor = BatchProcessor()
    completed_batch = processor.process_batch(batch)
    
    # Display results
    summary = completed_batch.get_processing_summary()
    print(f"✓ Batch {completed_batch.name} completed")
    print(f"✓ Videos processed: {summary['processed_videos']}/{summary['total_videos']}")
    print(f"✓ Success rate: {summary['progress_percentage']:.1f}%")
    
    return 0

if __name__ == '__main__':
    exit(main())