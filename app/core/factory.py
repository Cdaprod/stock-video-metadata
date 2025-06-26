# app/core/factory.py

from app.core.artifacts.video import VideoArtifact
from app.core.artifacts.batch import BatchArtifact

class ArtifactFactory:
    @staticmethod
    def create_video_from_path(file_path: str) -> VideoArtifact:
        # ...
        pass

    # ... all your create_* staticmethods