"""
Vision Cap API Server - Modular Architecture
"""
import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .core.logging_config import setup_logging
from .core.models.factory import ModelFactory
from .services.storage_service import StorageService
from .services.search_service import SearchService
from .api.routes.search import create_search_router
from .api.routes.feed import create_feed_router
from .api.routes.stats import create_stats_router

# Load settings
settings = get_settings()

# Setup logging
logger = setup_logging(
    log_dir=settings.log_dir,
    log_file="api.log",
    log_level=settings.log_level,
    max_bytes=settings.log_max_bytes,
    backup_count=settings.log_backup_count
)

# Global services (initialized at startup)
search_service: SearchService = None
storage_service: StorageService = None
face_model = None
embedding_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global search_service, storage_service, face_model, embedding_model
    
    # Startup
    logger.info("Starting Vision Cap API...")
    logger.info("Loading API models (face detection + text search only)...")
    
    # Initialize models
    face_model = ModelFactory.create_face_detection_model()
    embedding_model = ModelFactory.create_embedding_model()
    
    # Initialize services
    storage_service = StorageService()
    search_service = SearchService(
        face_model=face_model,
        embedding_model=embedding_model,
        storage_service=storage_service
    )
    
    # Register routes
    app.include_router(create_search_router(search_service))
    app.include_router(create_feed_router(storage_service))
    app.include_router(create_stats_router(storage_service))
    logger.info("API routes registered")
    
    logger.info("API startup complete")
    yield
    
    # Shutdown
    logger.info("API shutting down")

# Create FastAPI app
app = FastAPI(
    title="Vision Cap API",
    description="Local Event Photo Search System",
    version="2.0.0",
    lifespan=lifespan
)

# Mount static files
# Mount from parent images directory to serve both processed and thumbnails
if os.path.exists("/app/images"):
    images_dir = "/app/images"
elif os.path.exists(settings.processed_dir):
    # Get parent directory (images/) from processed_dir (images/processed)
    images_dir = str(Path(settings.processed_dir).parent)
else:
    images_dir = None

if images_dir and os.path.exists(images_dir):
    app.mount("/images", StaticFiles(directory=images_dir), name="images")
    logger.info(f"Mounted static files from: {images_dir}")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check routes
@app.get("/")
async def root():
    return {"message": "Vision Cap API", "status": "running", "version": "2.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy", "qdrant": "connected"}


if __name__ == "__main__":
    import uvicorn
    
    # Pass app object directly instead of string path to avoid import issues
    uvicorn.run(
        app,  # Pass the app object directly
        host=settings.api_host,
        port=settings.api_port,
        reload=False  # Disable reload when passing app object directly
    )

