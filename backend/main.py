from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import cv2
import numpy as np
from PIL import Image
import io
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from datetime import datetime

from models import load_models, get_face_embedding, get_clip_embedding

# Initialize Qdrant client
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

COLLECTION_IMAGES = "images"
COLLECTION_FACES = "faces"

# Global model instances (loaded at startup)
face_app = None
clip_model = None

# Ensure collections exist
def ensure_collections():
    try:
        # Check if collections exist, create if not
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if COLLECTION_IMAGES not in collection_names:
            qdrant_client.create_collection(
                collection_name=COLLECTION_IMAGES,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )
        
        if COLLECTION_FACES not in collection_names:
            qdrant_client.create_collection(
                collection_name=COLLECTION_FACES,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )
    except Exception as e:
        print(f"Error ensuring collections: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models and ensure collections
    global face_app, clip_model
    face_app, clip_model = load_models()
    ensure_collections()
    yield
    # Shutdown (if needed)
    pass

app = FastAPI(title="Vision Cap API", lifespan=lifespan)

# Mount static files for images
if os.path.exists("/app/images"):
    app.mount("/images", StaticFiles(directory="/app/images"), name="images")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Qdrant client
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


class SearchResponse(BaseModel):
    photo_id: str
    image_url: str
    thumbnail_url: str
    caption: Optional[str] = None
    tags: Optional[List[str]] = None
    face_count: Optional[int] = None
    similarity_score: Optional[float] = None

@app.get("/")
async def root():
    return {"message": "Vision Cap API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "qdrant": "connected"}

@app.post("/search/face", response_model=List[SearchResponse])
async def search_by_face(file: UploadFile = File(...), threshold: float = Query(0.6, ge=0.0, le=1.0)):
    """
    Search for photos by uploading a selfie/face image.
    Returns photos containing matching faces.
    """
    try:
        # Read and process image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Get face embedding
        faces = face_app.get(img)
        if not faces:
            return []
        
        # Use the first face found
        face_embedding = faces[0].embedding
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_FACES,
            query_vector=face_embedding,
            limit=20,
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
            image_points = qdrant_client.retrieve(
                collection_name=COLLECTION_IMAGES,
                ids=[photo_id]
            )
            
            if image_points:
                payload = image_points[0].payload
                results.append(SearchResponse(
                    photo_id=photo_id,
                    image_url=f"/images/processed/{photo_id}.jpg",
                    thumbnail_url=f"/images/thumbnails/{photo_id}.jpg",
                    caption=payload.get("caption"),
                    tags=payload.get("tags", []),
                    face_count=payload.get("face_count"),
                    similarity_score=score
                ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/text", response_model=List[SearchResponse])
async def search_by_text(query: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100)):
    """
    Search for photos by text description using CLIP.
    """
    try:
        # Get text embedding
        text_embedding = clip_model.encode([query])[0].tolist()
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_IMAGES,
            query_vector=text_embedding,
            limit=limit
        )
        
        results = []
        for result in search_results:
            payload = result.payload
            results.append(SearchResponse(
                photo_id=result.id,
                image_url=f"/images/processed/{result.id}.jpg",
                thumbnail_url=f"/images/thumbnails/{result.id}.jpg",
                caption=payload.get("caption"),
                tags=payload.get("tags", []),
                face_count=payload.get("face_count"),
                similarity_score=result.score
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feed", response_model=List[SearchResponse])
async def get_feed(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)):
    """
    Get paginated feed of all images.
    """
    try:
        # Get all points with pagination
        offset = (page - 1) * page_size
        
        scroll_result = qdrant_client.scroll(
            collection_name=COLLECTION_IMAGES,
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """
    Get statistics about the photo collection.
    """
    try:
        image_count = qdrant_client.get_collection(COLLECTION_IMAGES).points_count
        face_count = qdrant_client.get_collection(COLLECTION_FACES).points_count
        
        return {
            "images_processed": image_count,
            "unique_faces": face_count,
            "status": "active"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

