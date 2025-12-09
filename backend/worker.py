"""
Image processing worker that watches for new images and processes them.
Uses watchdog to monitor the /images/raw directory.
"""
import os
import time
import shutil
import uuid
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import cv2
from PIL import Image
import logging
from logging.handlers import RotatingFileHandler
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from datetime import datetime

from models import load_models, get_face_embedding, get_clip_embedding, get_florence_tags_and_caption, load_image_as_cv2

# Configuration
RAW_DIR = "/app/images/raw"
PROCESSING_DIR = "/app/images/processing"
PROCESSED_DIR = "/app/images/processed"
THUMBNAIL_DIR = "/app/images/thumbnails"

# Qdrant setup
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

COLLECTION_IMAGES = "images"
COLLECTION_FACES = "faces"

# Setup logging
LOG_DIR = os.getenv("LOG_DIR", "/app/logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            os.path.join(LOG_DIR, 'worker.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

# Load models once
face_app, clip_model, florence_processor, florence_model = load_models()

def ensure_directories():
    """Create necessary directories if they don't exist."""
    for dir_path in [RAW_DIR, PROCESSING_DIR, PROCESSED_DIR, THUMBNAIL_DIR]:
        os.makedirs(dir_path, exist_ok=True)

def ensure_collections():
    """Ensure Qdrant collections exist."""
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if COLLECTION_IMAGES not in collection_names:
            qdrant_client.create_collection(
                collection_name=COLLECTION_IMAGES,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )
            logger.info(f"Created collection: {COLLECTION_IMAGES}")
            # Create full-text indexes for hybrid search
            try:
                qdrant_client.create_payload_index(
                    collection_name=COLLECTION_IMAGES,
                    field_name="tags",
                    field_schema="keyword"
                )
                qdrant_client.create_payload_index(
                    collection_name=COLLECTION_IMAGES,
                    field_name="caption",
                    field_schema="text"
                )
                logger.info("Created full-text indexes on tags and caption")
            except Exception as e:
                logger.warning(f"Could not create full-text indexes (may already exist): {e}")
        
        if COLLECTION_FACES not in collection_names:
            qdrant_client.create_collection(
                collection_name=COLLECTION_FACES,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )
            logger.info(f"Created collection: {COLLECTION_FACES}")
    except Exception as e:
        logger.error(f"Error ensuring collections: {e}", exc_info=True)

def create_thumbnail(image_path, output_path, max_size=300):
    """Create a thumbnail of the image. Handles HEIC files."""
    try:
        img = Image.open(image_path)
        # Convert to RGB if needed (HEIC might be RGBA or other modes)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        img.save(output_path, "JPEG", quality=85)
    except Exception as e:
        logger.error(f"Error creating thumbnail for {image_path}: {e}", exc_info=True)

def process_image(image_path):
    """
    Process a single image:
    1. Extract face embeddings
    2. Extract CLIP embedding
    3. Generate metadata
    4. Store in Qdrant
    """
    try:
        logger.info(f"Processing: {image_path}")
        start_time = time.time()
        
        # Generate unique photo ID
        photo_id = str(uuid.uuid4())
        
        # Read image (handles HEIC files)
        img = load_image_as_cv2(image_path)
        if img is None:
            logger.error(f"Could not read image {image_path}")
            return False
        
        # Step 1: Face Detection and Embedding
        face_embeddings = get_face_embedding(img)
        face_count = len(face_embeddings)
        
        # Determine photo type
        if face_count == 0:
            photo_type = "no_faces"
        elif face_count == 1:
            photo_type = "solo"
        elif face_count == 2:
            photo_type = "duo"
        else:
            photo_type = "group"
        
        # Step 2: CLIP Semantic Embedding (normalized)
        clip_embedding = get_clip_embedding(image_path)
        
        # Step 3: Extract tags and caption using Florence-2 VLM
        florence_tags, florence_caption = get_florence_tags_and_caption(image_path)
        
        # Combine tags: photo type + Florence-2 detected objects
        tags = [photo_type]
        if florence_tags:
            tags.extend(florence_tags[:20])  # Limit to 20 tags max
        
        # Use Florence caption if available, otherwise fallback
        if florence_caption:
            caption = florence_caption
        else:
            caption = f"Photo with {face_count} face(s)"
        
        # Step 4: Store image in Qdrant
        image_point = PointStruct(
            id=photo_id,
            vector=clip_embedding.tolist(),
            payload={
                "photo_id": photo_id,
                "file_path": image_path,
                "face_count": face_count,
                "photo_type": photo_type,
                "tags": tags,
                "caption": caption,
                "processed_at": datetime.now().isoformat()
            }
        )
        
        qdrant_client.upsert(
            collection_name=COLLECTION_IMAGES,
            points=[image_point]
        )
        
        # Step 5: Store each face embedding
        # Qdrant requires point IDs to be either integers or valid UUIDs (no underscores)
        face_points = []
        for i, face_emb in enumerate(face_embeddings):
            # Generate a unique UUID for each face
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
        
        if face_points:
            qdrant_client.upsert(
                collection_name=COLLECTION_FACES,
                points=face_points
            )
        
        # Step 6: Move and save processed image (convert HEIC to JPG)
        processed_path = os.path.join(PROCESSED_DIR, f"{photo_id}.jpg")
        # If it's a HEIC file, convert to JPG; otherwise copy as-is
        ext = os.path.splitext(image_path)[1].lower()
        if ext in ['.heic', '.heif']:
            # Convert HEIC to JPG using PIL
            pil_img = Image.open(image_path)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            pil_img.save(processed_path, "JPEG", quality=95)
        else:
            shutil.copy2(image_path, processed_path)
        
        # Create thumbnail
        thumbnail_path = os.path.join(THUMBNAIL_DIR, f"{photo_id}.jpg")
        create_thumbnail(image_path, thumbnail_path)
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Processed {photo_id} in {elapsed:.2f}s ({face_count} faces)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}", exc_info=True)
        return False

class ImageHandler(FileSystemEventHandler):
    """Handler for file system events."""
    
    def __init__(self):
        self.processing = set()
    
    def on_created(self, event):
        """Called when a new file is created."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        
        # Only process image files (including HEIC)
        valid_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.JPG', '.JPEG', '.PNG', '.HEIC', '.HEIF'}
        if not any(file_path.lower().endswith(ext.lower()) for ext in valid_extensions):
            return
        
        # Avoid processing the same file twice
        if file_path in self.processing:
            return
        
        self.processing.add(file_path)
        
        # Wait a bit to ensure file is fully written
        time.sleep(0.5)
        
        # Move to processing directory
        try:
            processing_path = os.path.join(PROCESSING_DIR, os.path.basename(file_path))
            shutil.move(file_path, processing_path)
            
            # Process the image
            if process_image(processing_path):
                # Delete from processing after successful processing
                os.remove(processing_path)
            else:
                # Move back to raw if processing failed
                shutil.move(processing_path, file_path)
            
            self.processing.discard(file_path)
            
        except Exception as e:
            logger.error(f"Error handling file {file_path}: {e}", exc_info=True)
            self.processing.discard(file_path)

def main():
    """Main worker loop."""
    logger.info("Starting Vision Cap Worker...")
    
    ensure_directories()
    ensure_collections()
    
    # Set up watchdog
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, RAW_DIR, recursive=False)
    observer.start()
    
    logger.info(f"Watching directory: {RAW_DIR}")
    logger.info("Worker is running. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Worker stopped.")
    
    observer.join()

if __name__ == "__main__":
    main()

