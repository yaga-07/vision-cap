"""
Feed API routes.
"""
import logging
from fastapi import APIRouter, HTTPException, Query
from typing import List

from ...api.schemas import SearchResponse
from ...services.storage_service import StorageService

logger = logging.getLogger(__name__)

def create_feed_router(storage_service: StorageService) -> APIRouter:
    """Create feed router with dependencies."""
    router = APIRouter(prefix="/feed", tags=["feed"])
    
    @router.get("", response_model=List[SearchResponse])
    async def get_feed(
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100)
    ):
        """Get paginated feed of all images."""
        try:
            offset = (page - 1) * page_size
            
            scroll_result = storage_service.scroll(
                collection=storage_service.collection_images,
                limit=page_size,
                offset=offset,
                with_payload=True
            )
            
            results = []
            for point in scroll_result[0]:
                payload = point.payload
                results.append(SearchResponse(
                    photo_id=point.id,
                    image_url=f"/images/processed/{point.id}.jpg",
                    thumbnail_url=f"/images/thumbnails/{point.id}.jpg",
                    caption=payload.get("caption"),
                    tags=payload.get("tags", []),
                    face_count=payload.get("face_count")
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting feed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    return router

