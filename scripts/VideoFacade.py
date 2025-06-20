# models/video_facade.py
from proxy import MediaProxyArtifact    
from pydantic import BaseModel, Field
from typing import Dict, Any
from VideoArtifact import VideoArtifact

class VideoFacade(BaseModel):
    artifact: VideoArtifact
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "extra": "allow",
        "arbitrary_types_allowed": True,
    }

    # Dynamic fields...
    def register_field(self, key: str, value: Any):
        self.metadata[key] = value
        setattr(self, key, value)

    def to_proxy(self) -> MediaProxyArtifact:
        """Return a Pydantic proxy object (strongly typed)."""
        return MediaProxyArtifact(
            id=self.artifact.id,
            filename=self.artifact.filename,
            source_type=self.artifact.source_type,
            created_at=self.artifact.created_at,
            state=self.artifact.state.value,
            file_path=self.artifact.file_path,
            duration=self.artifact.duration,
            resolution=list(self.artifact.resolution) if self.artifact.resolution else None,
            codec=self.artifact.codec,
            bitrate=self.artifact.bitrate,
            frame_rate=self.artifact.frame_rate,
            metadata=self.metadata,
            events=self.artifact.get_history(),
        )
        
    def __getitem__(self, key):
        return self.metadata.get(key, getattr(self, key, None))

    def register_fields(self, data: Dict[str, Any]):
        for k, v in data.items():
            self.register_field(k, v)