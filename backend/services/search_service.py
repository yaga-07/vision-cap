"""
Search service for image and face search.
"""
import re
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from ..core.models.base import BaseFaceDetectionModel, BaseEmbeddingModel
from ..services.storage_service import StorageService
from ..utils.image_utils import load_image_as_cv2
from qdrant_client.models import Filter, FieldCondition, MatchText

logger = logging.getLogger(__name__)

class SearchService:
    """Service for search operations."""
    
    def __init__(
        self,
        face_model: BaseFaceDetectionModel,
        embedding_model: BaseEmbeddingModel,
        storage_service: Optional[StorageService] = None
    ):
        self.face_model = face_model
        self.embedding_model = embedding_model
        self.storage = storage_service or StorageService()
    
    def normalize_similarity_score(self, score: float) -> float:
        """
        Normalize similarity score to 0-1 range.
        
        Args:
            score: Raw similarity score
        
        Returns:
            Normalized score
        """
        # Normalize from [-1, 1] to [0, 1]
        normalized = (score + 1) / 2
        # Apply power curve to emphasize higher scores
        return normalized ** 0.7
    
    def build_text_filter(self, query: str) -> Optional[Filter]:
        """
        Build Qdrant filter for full-text search.
        
        Args:
            query: Search query string
        
        Returns:
            Qdrant Filter or None
        """
        # Extract keywords
        words = re.findall(r'\b\w+\b', query.lower())
        stop_words = {
            'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
            'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        if not keywords:
            keywords = [query.lower()]
        
        # Build filter conditions
        conditions = []
        for keyword in keywords:
            conditions.append(
                FieldCondition(key="tags", match=MatchText(text=keyword))
            )
            conditions.append(
                FieldCondition(key="caption", match=MatchText(text=keyword))
            )
        
        if conditions:
            return Filter(should=conditions)
        return None
    
    def search_by_face(
        self,
        image: np.ndarray,
        threshold: float = 0.6,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for photos by face.
        
        Args:
            image: Face image as numpy array
            threshold: Similarity threshold
            limit: Maximum results
        
        Returns:
            List of search results
        """
        try:
            # Detect faces using the model
            faces = self.face_model.detect_faces(image)
            if not faces:
                return []
            
            # Use first face embedding
            face_embedding = faces[0]
            
            # Search in Qdrant
            search_results = self.storage.search_by_vector(
                query_vector=face_embedding.tolist(),
                collection=self.storage.collection_faces,
                limit=limit,
                score_threshold=threshold
            )
            
            # Get unique photo IDs
            photo_ids = {}
            for result in search_results:
                photo_id = result.payload.get("photo_id")
                if photo_id and photo_id not in photo_ids:
                    photo_ids[photo_id] = result.score
            
            # Fetch full image data
            results = []
            for photo_id, score in photo_ids.items():
                image_points = self.storage.retrieve_by_ids(
                    ids=[photo_id],
                    collection=self.storage.collection_images
                )
                
                if image_points:
                    payload = image_points[0].payload
                    results.append({
                        "photo_id": photo_id,
                        "payload": payload,
                        "similarity_score": score
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in face search: {e}", exc_info=True)
            raise
    
    def search_by_text(
        self,
        query: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for photos by text query.
        
        Args:
            query: Search query string
            limit: Maximum results
        
        Returns:
            List of search results
        """
        try:
            # Get text embedding
            text_embedding = self.embedding_model.encode_text(query)
            
            # Build text filter
            text_filter = self.build_text_filter(query)
            
            # Search in Qdrant
            search_results = self.storage.search_by_vector(
                query_vector=text_embedding.tolist(),
                collection=self.storage.collection_images,
                limit=limit * 2,  # Get more for re-ranking
                query_filter=text_filter
            )
            
            # Process results
            results = []
            for result in search_results[:limit]:
                normalized_score = self.normalize_similarity_score(result.score)
                results.append({
                    "photo_id": result.id,
                    "payload": result.payload,
                    "similarity_score": normalized_score
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in text search: {e}", exc_info=True)
            # Fallback to vector-only search
            try:
                logger.warning("Falling back to vector-only search")
                text_embedding = self.embedding_model.encode_text(query)
                search_results = self.storage.search_by_vector(
                    query_vector=text_embedding.tolist(),
                    collection=self.storage.collection_images,
                    limit=limit
                )
                
                results = []
                for result in search_results:
                    normalized_score = self.normalize_similarity_score(result.score)
                    results.append({
                        "photo_id": result.id,
                        "payload": result.payload,
                        "similarity_score": normalized_score
                    })
                return results
            except Exception as e2:
                logger.error(f"Fallback search failed: {e2}", exc_info=True)
                raise
    
    def search_similar(
        self,
        photo_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Find similar images to a given photo.
        
        Args:
            photo_id: Photo ID to find similar images for
            limit: Maximum results
        
        Returns:
            List of similar images
        """
        try:
            # Retrieve image embedding
            image_points = self.storage.retrieve_by_ids(
                ids=[photo_id],
                collection=self.storage.collection_images,
                with_vectors=True
            )
            
            if not image_points:
                return []
            
            query_vector = image_points[0].vector
            
            # Search for similar images
            search_results = self.storage.search_by_vector(
                query_vector=query_vector,
                collection=self.storage.collection_images,
                limit=limit + 1
            )
            
            # Filter out the original image
            results = []
            for result in search_results:
                if result.id == photo_id:
                    continue
                
                normalized_score = self.normalize_similarity_score(result.score)
                results.append({
                    "photo_id": result.id,
                    "payload": result.payload,
                    "similarity_score": normalized_score
                })
                
                if len(results) >= limit:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similar image search: {e}", exc_info=True)
            raise

