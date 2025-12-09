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
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from datetime import datetime

from models import load_models, get_face_embedding, get_clip_embedding

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

# Load models once
face_app, clip_model = load_models()

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
            print(f"Created collection: {COLLECTION_IMAGES}")
        
        if COLLECTION_FACES not in collection_names:
            qdrant_client.create_collection(
                collection_name=COLLECTION_FACES,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )
            print(f"Created collection: {COLLECTION_FACES}")
    except Exception as e:
        print(f"Error ensuring collections: {e}")

def create_thumbnail(image_path, output_path, max_size=300):
    """Create a thumbnail of the image."""
    try:
        img = Image.open(image_path)
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        img.save(output_path, "JPEG", quality=85)
    except Exception as e:
        print(f"Error creating thumbnail: {e}")

def process_image(image_path):
    """
    Process a single image:
    1. Extract face embeddings
    2. Extract CLIP embedding
    3. Generate metadata
    4. Store in Qdrant
    """
    try:
        print(f"Processing: {image_path}")
        start_time = time.time()
        
        # Generate unique photo ID
        photo_id = str(uuid.uuid4())
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
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
        
        # Step 2: CLIP Semantic Embedding
        clip_embedding = get_clip_embedding(image_path)
        
        # Step 3: Basic metadata (Florence-2 can be added later for MVP speed)
        # For now, use simple tags based on face count
        tags = [photo_type]
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
        
        # Step 6: Move and save processed image
        processed_path = os.path.join(PROCESSED_DIR, f"{photo_id}.jpg")
        shutil.copy2(image_path, processed_path)
        
        # Create thumbnail
        thumbnail_path = os.path.join(THUMBNAIL_DIR, f"{photo_id}.jpg")
        create_thumbnail(image_path, thumbnail_path)
        
        elapsed = time.time() - start_time
        print(f"✓ Processed {photo_id} in {elapsed:.2f}s ({face_count} faces)")
        
        return True
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        import traceback
        traceback.print_exc()
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
        
        # Only process image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
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
            print(f"Error handling file {file_path}: {e}")
            self.processing.discard(file_path)

def main():
    """Main worker loop."""
    print("Starting Vision Cap Worker...")
    
    ensure_directories()
    ensure_collections()
    
    # Set up watchdog
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, RAW_DIR, recursive=False)
    observer.start()
    
    print(f"Watching directory: {RAW_DIR}")
    print("Worker is running. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nWorker stopped.")
    
    observer.join()

if __name__ == "__main__":
    main()

