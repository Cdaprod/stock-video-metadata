# app/core/artifacts/base.py
from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Any, Generic, TypeVar
from enum import Enum, auto
from pydantic.dataclasses import dataclass

class ArtifactState(str, Enum):
    CREATED = "created"
    VALIDATED = "validated"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"

class ArtifactEventType(str, Enum):
    CREATED  = "created"
    SOURCE_ATTACHED = "source_attached"
    DATA_ATTACHED   = "data_attached"
    METADATA_EXTRACTED = "metadata_extracted"
    PROCESSING_STARTED = "processing_started"
    PROCESSING_COMPLETED = "processing_completed"
    PROCESSING_FAILED = "processing_failed"
    # …extend as needed …

@dataclass
class ArtifactEvent:
    type: ArtifactEventType
    data: Dict[str, Any] | None = None
    timestamp: datetime = datetime.now()

TMeta = TypeVar("TMeta", bound=Dict[str, Any])

@dataclass
class MetadataMixin(Generic[TMeta]):
    metadata: TMeta

    def register_field(self, key: str, value: Any) -> None:
        self.metadata[key] = value  # type: ignore[index]

    def register_fields(self, data: Dict[str, Any]) -> None:
        for k, v in data.items():
            self.register_field(k, v)

@dataclass
class Artifact(MetadataMixin[Dict[str, Any]]):
    id: str
    created_at: datetime = datetime.now()
    state: ArtifactState = ArtifactState.CREATED
    events: List[ArtifactEvent] = None
    version: int = 1

    def __post_init__(self):
        self.events = self.events or []

    # --- event helpers --------------------------------------------------- #
    def emit(self, event_type: ArtifactEventType, data: Dict[str, Any] | None = None):
        evt = ArtifactEvent(type=event_type, data=data or {})
        self.events.append(evt)
        self.version += 1
        self._apply(evt)

    def _apply(self, event: ArtifactEvent):  # pragma: no cover
        """Override in subclasses to mutate state."""
        pass