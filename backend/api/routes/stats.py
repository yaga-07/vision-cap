"""
Stats API routes.
"""
import logging
from fastapi import APIRouter

from ...api.schemas import StatsResponse
from ...services.storage_service import StorageService

logger = logging.getLogger(__name__)

def create_stats_router(storage_service: StorageService) -> APIRouter:
    """Create stats router with dependencies."""
    router = APIRouter(prefix="/stats", tags=["stats"])
    
    @router.get("", response_model=StatsResponse)
    async def get_stats():
        """Get statistics about the photo collection."""
        try:
            image_stats = storage_service.get_collection_stats(
                storage_service.collection_images
            )
            face_stats = storage_service.get_collection_stats(
                storage_service.collection_faces
            )
            
            return StatsResponse(
                images_processed=image_stats.get("points_count", 0),
                unique_faces=face_stats.get("points_count", 0),
                status="active"
            )
        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            return StatsResponse(
                images_processed=0,
                unique_faces=0,
                status="error"
            )
    
    return router

