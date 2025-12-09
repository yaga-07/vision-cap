"""
Storage service for Qdrant operations.
"""
import logging
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchText
)
from uuid import UUID

from ..config import get_settings

logger = logging.getLogger(__name__)

class StorageService:
    """Service for Qdrant database operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = QdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port
        )
        self.collection_images = self.settings.collection_images
        self.collection_faces = self.settings.collection_faces
        self._ensure_collections()
    
    def _ensure_collections(self):
        """Ensure Qdrant collections exist with proper configuration."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            # Create images collection if needed
            if self.collection_images not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_images,
                    vectors_config=VectorParams(
                        size=self.settings.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_images}")
                
                # Create full-text indexes
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_images,
                        field_name="tags",
                        field_schema="keyword"
                    )
                    self.client.create_payload_index(
                        collection_name=self.collection_images,
                        field_name="caption",
                        field_schema="text"
                    )
                    logger.info("Created full-text indexes on tags and caption")
                except Exception as e:
                    logger.warning(f"Could not create full-text indexes: {e}")
            
            # Create faces collection if needed
            if self.collection_faces not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_faces,
                    vectors_config=VectorParams(
                        size=self.settings.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_faces}")
        except Exception as e:
            logger.error(f"Error ensuring collections: {e}", exc_info=True)
    
    def store_image(
        self,
        photo_id: str,
        embedding: List[float],
        payload: Dict[str, Any]
    ) -> bool:
        """
        Store an image in Qdrant.
        
        Args:
            photo_id: Unique photo identifier
            embedding: Image embedding vector
            payload: Metadata payload
        
        Returns:
            True if successful
        """
        try:
            point = PointStruct(
                id=photo_id,
                vector=embedding,
                payload=payload
            )
            self.client.upsert(
                collection_name=self.collection_images,
                points=[point]
            )
            return True
        except Exception as e:
            logger.error(f"Error storing image: {e}", exc_info=True)
            return False
    
    def store_faces(
        self,
        face_points: List[PointStruct]
    ) -> bool:
        """
        Store face embeddings in Qdrant.
        
        Args:
            face_points: List of PointStruct objects for faces
        
        Returns:
            True if successful
        """
        try:
            if face_points:
                self.client.upsert(
                    collection_name=self.collection_faces,
                    points=face_points
                )
            return True
        except Exception as e:
            logger.error(f"Error storing faces: {e}", exc_info=True)
            return False
    
    def search_by_vector(
        self,
        query_vector: List[float],
        collection: str,
        limit: int = 20,
        score_threshold: Optional[float] = None,
        query_filter: Optional[Filter] = None
    ) -> List:
        """
        Search by vector similarity.
        
        Args:
            query_vector: Query embedding vector
            collection: Collection name
            limit: Maximum results
            score_threshold: Minimum similarity score
            query_filter: Optional filter
        
        Returns:
            List of search results
        """
        try:
            return self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter
            )
        except Exception as e:
            logger.error(f"Error in vector search: {e}", exc_info=True)
            raise
    
    def retrieve_by_ids(
        self,
        ids: List[str],
        collection: str,
        with_vectors: bool = False
    ) -> List:
        """
        Retrieve points by IDs.
        
        Args:
            ids: List of point IDs
            collection: Collection name
            with_vectors: Include vectors in response
        
        Returns:
            List of points
        """
        try:
            return self.client.retrieve(
                collection_name=collection,
                ids=ids,
                with_vectors=with_vectors
            )
        except Exception as e:
            logger.error(f"Error retrieving points: {e}", exc_info=True)
            raise
    
    def scroll(
        self,
        collection: str,
        limit: int = 20,
        offset: int = 0,
        with_payload: bool = True
    ) -> tuple:
        """
        Scroll through collection.
        
        Args:
            collection: Collection name
            limit: Number of results
            offset: Offset for pagination
            with_payload: Include payload
        
        Returns:
            Tuple of (points, next_page_offset)
        """
        try:
            return self.client.scroll(
                collection_name=collection,
                limit=limit,
                offset=offset,
                with_payload=with_payload
            )
        except Exception as e:
            logger.error(f"Error scrolling collection: {e}", exc_info=True)
            raise
    
    def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Args:
            collection: Collection name
        
        Returns:
            Dictionary with collection stats
        """
        try:
            collection_info = self.client.get_collection(collection)
            return {
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}", exc_info=True)
            return {}

