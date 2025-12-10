"""
Image processing worker - Modular Architecture
"""
import os
import time
import shutil
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .config import get_settings
from .core.logging_config import setup_logging
from .core.models.factory import ModelFactory
from .services.storage_service import StorageService
from .services.image_processor import ImageProcessor

# Load settings
settings = get_settings()

# Setup logging
logger = setup_logging(
    log_dir=settings.log_dir,
    log_file="worker.log",
    log_level=settings.log_level,
    max_bytes=settings.log_max_bytes,
    backup_count=settings.log_backup_count
)

class ImageHandler(FileSystemEventHandler):
    """Handler for file system events."""
    
    def __init__(self, image_processor: ImageProcessor):
        self.image_processor = image_processor
        self.processing = set()
    
    def on_created(self, event):
        """Called when a new file is created."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        
        # Only process image files (including HEIC)
        valid_extensions = {
            '.jpg', '.jpeg', '.png', '.heic', '.heif',
            '.JPG', '.JPEG', '.PNG', '.HEIC', '.HEIF'
        }
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
            processing_path = os.path.join(settings.processing_dir, os.path.basename(file_path))
            shutil.move(file_path, processing_path)
            
            # Process the image
            success, photo_id = self.image_processor.process_image(processing_path)
            
            if success:
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
    
    # Ensure directories exist
    for dir_path in [
        settings.raw_dir,
        settings.processing_dir,
        settings.processed_dir,
        settings.thumbnail_dir
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Initialize models
    logger.info("Loading models...")
    face_model = ModelFactory.create_face_detection_model()
    embedding_model = ModelFactory.create_embedding_model()
    vlm = ModelFactory.create_vlm(
        model_type=settings.vlm_model,
        model_name=settings.google_genai_model,
        api_key=settings.google_api_key
    )  # May return None if loading fails
    
    # Initialize services
    storage_service = StorageService()
    image_processor = ImageProcessor(
        face_model=face_model,
        embedding_model=embedding_model,
        vlm=vlm,
        storage_service=storage_service
    )
    
    # Set up watchdog
    event_handler = ImageHandler(image_processor)
    observer = Observer()
    observer.schedule(event_handler, settings.raw_dir, recursive=False)
    observer.start()
    
    logger.info(f"Watching directory: {settings.raw_dir}")
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

