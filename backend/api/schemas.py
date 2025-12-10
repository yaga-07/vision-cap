"""
Pydantic schemas for API requests and responses.
"""
from pydantic import BaseModel
from typing import List, Optional

class SearchResponse(BaseModel):
    """Response model for search results."""
    photo_id: str
    image_url: str
    thumbnail_url: str
    caption: Optional[str] = None  # Legacy field, kept for backward compatibility
    tags: Optional[List[str]] = None
    face_count: Optional[int] = None
    similarity_score: Optional[float] = None
    # Note: generic_text and photographer_text are kept internal for search only

class StatsResponse(BaseModel):
    """Response model for statistics."""
    images_processed: int
    unique_faces: int
    status: str

