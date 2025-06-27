# app/services/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class ArtifactService(ABC):
    """Domain-agnostic service interface for persistence & lifecycle."""
    @abstractmethod
    def ingest_uploads(self, uploads: List[Dict[str, Any]], **kw) -> str: ...
    @abstractmethod
    def ingest_paths(self, paths: List[str], **kw) -> str: ...
    @abstractmethod
    def list(self) -> List[Dict[str, str]]: ...
    @abstractmethod
    def get(self, artifact_id: str) -> Dict[str, Any]: ...
    @abstractmethod
    def manifest(self, artifact_id: str) -> Dict[str, Any]: ...

