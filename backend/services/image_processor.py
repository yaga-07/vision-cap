"""
Image processing service.
"""
import os
import uuid
import shutil
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from ..core.models.base import BaseFaceDetectionModel, BaseEmbeddingModel, BaseVLM
from ..services.storage_service import StorageService
from ..utils.image_utils import load_image_as_cv2, create_thumbnail, convert_heic_to_jpg
from ..config import get_settings
from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Service for processing images."""
    
    def __init__(
        self,
        face_model: BaseFaceDetectionModel,
        embedding_model: BaseEmbeddingModel,
        vlm: Optional[BaseVLM] = None,
        storage_service: Optional[StorageService] = None
    ):
        self.face_model = face_model
        self.embedding_model = embedding_model
        self.vlm = vlm
        self.storage = storage_service or StorageService()
        self.settings = get_settings()
    
    def process_image(self, image_path: str) -> Tuple[bool, Optional[str]]:
        """
        Process a single image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Tuple of (success, photo_id)
        """
        try:
            logger.info(f"Processing: {image_path}")
            start_time = datetime.now()
            
            # Generate unique photo ID
            photo_id = str(uuid.uuid4())
            
            # Read image
            img_cv2 = load_image_as_cv2(image_path)
            if img_cv2 is None:
                logger.error(f"Could not read image {image_path}")
                return False, None
            
            # Step 1: Face Detection
            face_embeddings = self.face_model.detect_faces(img_cv2)
            face_count = len(face_embeddings)
            
            # Determine photo type
            photo_type = self._determine_photo_type(face_count)
            
            # Step 2: Image Embedding
            image_embedding = self.embedding_model.encode_image(image_path)
            
            # Step 3: Generate Tags and Caption (if VLM available)
            tags, caption = self._generate_metadata(image_path, face_count)
            
            # Combine tags
            all_tags = [photo_type]
            if tags:
                all_tags.extend(tags[:20])
            
            # Use caption or fallback
            final_caption = caption or f"Photo with {face_count} face(s)"
            
            # Step 4: Store in Qdrant
            payload = {
                "photo_id": photo_id,
                "file_path": image_path,
                "face_count": face_count,
                "photo_type": photo_type,
                "tags": all_tags,
                "caption": final_caption,
                "processed_at": datetime.now().isoformat()
            }
            
            success = self.storage.store_image(
                photo_id=photo_id,
                embedding=image_embedding.tolist(),
                payload=payload
            )
            
            if not success:
                return False, None
            
            # Step 5: Store Face Embeddings
            face_points = []
            for i, face_emb in enumerate(face_embeddings):
                face_id = str(uuid.uuid4())
                face_point = PointStruct(
                    id=face_id,
                    vector=face_emb.tolist(),
                    payload={
                        "photo_id": photo_id,
                        "face_index": i,
                        "parent_image": photo_id
                    }
                )
                face_points.append(face_point)
            
            self.storage.store_faces(face_points)
            
            # Step 6: Save processed image and thumbnail
            self._save_processed_image(image_path, photo_id)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"✓ Processed {photo_id} in {elapsed:.2f}s ({face_count} faces)")
            
            return True, photo_id
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}", exc_info=True)
            return False, None
    
    def _determine_photo_type(self, face_count: int) -> str:
        """Determine photo type based on face count."""
        if face_count == 0:
            return "no_faces"
        elif face_count == 1:
            return "solo"
        elif face_count == 2:
            return "duo"
        else:
            return "group"
    
    def _generate_metadata(
        self,
        image_path: str,
        face_count: int
    ) -> Tuple[List[str], Optional[str]]:
        """Generate tags and caption using VLM."""
        if self.vlm is None:
            return [], None
        
        try:
            tags, caption = self.vlm.generate_tags_and_caption(image_path)
            return tags, caption
        except Exception as e:
            logger.warning(f"Error generating metadata: {e}")
            return [], None
    
    def _save_processed_image(self, image_path: str, photo_id: str):
        """Save processed image and create thumbnail."""
        try:
            # Determine if HEIC conversion needed
            ext = os.path.splitext(image_path)[1].lower()
            processed_path = Path(self.settings.processed_dir) / f"{photo_id}.jpg"
            
            if ext in ['.heic', '.heif']:
                convert_heic_to_jpg(image_path, str(processed_path))
            else:
                shutil.copy2(image_path, processed_path)
            
            # Create thumbnail
            thumbnail_path = Path(self.settings.thumbnail_dir) / f"{photo_id}.jpg"
            create_thumbnail(str(processed_path), str(thumbnail_path))
            
        except Exception as e:
            logger.error(f"Error saving processed image: {e}", exc_info=True)

