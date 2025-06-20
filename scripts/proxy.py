# scripts/proxy.py
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime

class MediaProxyArtifact(BaseModel):
    id: str
    filename: str
    source_type: str
    created_at: Optional[datetime] = None
    state: Optional[str] = "created"
    file_path: Optional[str] = None
    asset_url: Optional[HttpUrl] = None
    thumbnail_url: Optional[HttpUrl] = None
    duration: Optional[float] = None
    resolution: Optional[List[int]] = None
    codec: Optional[str] = None
    bitrate: Optional[int] = None
    frame_rate: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    events: Optional[List[Dict[str, Any]]] = None

    model_config = {
        "json_schema_extra": {},
        "str_strip_whitespace": True
    }