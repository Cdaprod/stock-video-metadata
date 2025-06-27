# app/services/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List

class ArtifactService(ABC):
    """Autonomous interface for persistence & lifecycle of any artifact."""

    @abstractmethod
    def ingest_uploads(self, uploads: List[Dict[str, Any]], **kwargs) -> str:
        """Ingest a list of byte-payloads, return newly created artifact ID."""
        raise NotImplementedError("ingest_uploads() must be implemented by subclass")

    @abstractmethod
    def ingest_paths(self, paths: List[str], **kwargs) -> str:
        """Ingest a list of filesystem paths, return newly created artifact ID."""
        raise NotImplementedError("ingest_paths() must be implemented by subclass")

    @abstractmethod
    def list(self) -> List[Dict[str, str]]:
        """List all known artifact IDs and (optional) names."""
        raise NotImplementedError("list() must be implemented by subclass")

    @abstractmethod
    def get(self, artifact_id: str) -> Dict[str, Any]:
        """Get a proxy-enhanced view of one artifact (or fallback info)."""
        raise NotImplementedError("get() must be implemented by subclass")

    @abstractmethod
    def manifest(self, artifact_id: str) -> Dict[str, Any]:
        """Get the raw stored manifest for one artifact."""
        raise NotImplementedError("manifest() must be implemented by subclass")

    def __call__(self, *args, **kwargs) -> str:
        """
        Allow service instances to be called directly:
            new_id = service(uploads=…, **kwargs)
        """
        return self.ingest_uploads(*args, **kwargs)