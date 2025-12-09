"""
Search API routes.
"""
import logging
import cv2
import numpy as np
from PIL import Image
import io
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from typing import List

from ...api.schemas import SearchResponse
from ...services.search_service import SearchService
from ...utils.image_utils import load_image_as_cv2

logger = logging.getLogger(__name__)

def create_search_router(search_service: SearchService) -> APIRouter:
    """Create search router with dependencies."""
    router = APIRouter(prefix="/search", tags=["search"])
    
    @router.post("/face", response_model=List[SearchResponse])
    async def search_by_face(
        file: UploadFile = File(...),
        threshold: float = Query(0.6, ge=0.0, le=1.0)
    ):
        """Search for photos by uploading a face image."""
        try:
            logger.info(f"Face search request received, threshold: {threshold}")
            
            # Read and process image
            contents = await file.read()
            
            # Handle HEIC files
            filename = file.filename.lower() if file.filename else ""
            is_heic = filename.endswith(('.heic', '.heif')) or contents[:12] == b'\x00\x00\x00 ftypheic'
            
            if is_heic:
                try:
                    from pillow_heif import register_heif_opener
                    register_heif_opener()
                except:
                    pass
                
                pil_img = Image.open(io.BytesIO(contents))
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                img_array = np.array(pil_img)
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                nparr = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            
            # Search
            results = search_service.search_by_face(img, threshold=threshold)
            
            if not results:
                return []
            
            # Format response
            formatted_results = []
            for result in results:
                payload = result["payload"]
                photo_id = result["photo_id"]
                formatted_results.append(SearchResponse(
                    photo_id=photo_id,
                    image_url=f"/images/processed/{photo_id}.jpg",
                    thumbnail_url=f"/images/thumbnails/{photo_id}.jpg",
                    caption=payload.get("caption"),
                    tags=payload.get("tags", []),
                    face_count=payload.get("face_count"),
                    similarity_score=result["similarity_score"]
                ))
            
            logger.info(f"Returning {len(formatted_results)} unique photos")
            return formatted_results
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in face search: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/text", response_model=List[SearchResponse])
    async def search_by_text(
        query: str = Query(..., min_length=1),
        limit: int = Query(20, ge=1, le=100)
    ):
        """Search for photos by text description."""
        try:
            logger.info(f"Text search query: '{query}'")
            
            # Search
            results = search_service.search_by_text(query, limit=limit)
            
            # Format response
            formatted_results = []
            for result in results:
                payload = result["payload"]
                photo_id = result["photo_id"]
                formatted_results.append(SearchResponse(
                    photo_id=photo_id,
                    image_url=f"/images/processed/{photo_id}.jpg",
                    thumbnail_url=f"/images/thumbnails/{photo_id}.jpg",
                    caption=payload.get("caption"),
                    tags=payload.get("tags", []),
                    face_count=payload.get("face_count"),
                    similarity_score=result["similarity_score"]
                ))
            
            logger.info(f"Returning {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in text search: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/similar", response_model=List[SearchResponse])
    async def search_similar(
        photo_id: str = Query(..., min_length=1),
        limit: int = Query(20, ge=1, le=100)
    ):
        """Find similar images to a given photo."""
        try:
            logger.info(f"Similar image search for photo_id: {photo_id}")
            
            # Search
            results = search_service.search_similar(photo_id, limit=limit)
            
            # Format response
            formatted_results = []
            for result in results:
                payload = result["payload"]
                photo_id_result = result["photo_id"]
                formatted_results.append(SearchResponse(
                    photo_id=photo_id_result,
                    image_url=f"/images/processed/{photo_id_result}.jpg",
                    thumbnail_url=f"/images/thumbnails/{photo_id_result}.jpg",
                    caption=payload.get("caption"),
                    tags=payload.get("tags", []),
                    face_count=payload.get("face_count"),
                    similarity_score=result["similarity_score"]
                ))
            
            logger.info(f"Found {len(formatted_results)} similar images")
            return formatted_results
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in similar image search: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    return router

