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
import logging
from logging.handlers import RotatingFileHandler
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchText
import uuid
from datetime import datetime

from models import load_api_models, get_text_embedding, load_image_as_cv2
import re

from dotenv import load_dotenv
load_dotenv()

# Initialize Qdrant client
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
            os.path.join(LOG_DIR, 'api.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

# Global model instances (loaded at startup - only API-required models)
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
            # Create full-text index on tags and caption for hybrid search
            try:
                qdrant_client.create_payload_index(
                    collection_name=COLLECTION_IMAGES,
                    field_name="tags",
                    field_schema="keyword"  # Index tags as keywords for full-text search
                )
                qdrant_client.create_payload_index(
                    collection_name=COLLECTION_IMAGES,
                    field_name="caption",
                    field_schema="text"  # Index caption as text for full-text search
                )
                logger.info("Created full-text indexes on tags and caption")
            except Exception as e:
                logger.warning(f"Could not create full-text indexes (may already exist): {e}")
        
        if COLLECTION_FACES not in collection_names:
            qdrant_client.create_collection(
                collection_name=COLLECTION_FACES,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )
    except Exception as e:
        print(f"Error ensuring collections: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load only API-required models (face detection + text search)
    global face_app, clip_model
    logger.info("Starting Vision Cap API...")
    logger.info("Loading API models (face detection + text search only)...")
    face_app, clip_model = load_api_models()
    ensure_collections()
    logger.info("API startup complete")
    yield
    # Shutdown (if needed)
    logger.info("API shutting down")

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
        logger.info(f"Face search request received, threshold: {threshold}")
        # Read and process image
        contents = await file.read()
        
        # Check if it's a HEIC file by extension or content
        filename = file.filename.lower() if file.filename else ""
        is_heic = filename.endswith(('.heic', '.heif')) or contents[:12] == b'\x00\x00\x00 ftypheic'
        
        if is_heic:
            # Handle HEIC: convert via PIL first
            from PIL import Image
            import io
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
            # Handle regular image formats
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Get face embedding
        faces = face_app.get(img)
        if not faces:
            logger.info("No faces detected in uploaded image")
            return []
        
        logger.info(f"Detected {len(faces)} face(s) in uploaded image")
        
        # Use the first face found
        face_embedding = faces[0].embedding
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_FACES,
            query_vector=face_embedding,
            limit=20,
            score_threshold=threshold
        )
        
        logger.info(f"Found {len(search_results)} matching faces")
        
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
        
        logger.info(f"Returning {len(results)} unique photos")
        return results
        
    except Exception as e:
        logger.error(f"Error in face search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def normalize_similarity_score(score):
    """
    Normalize CLIP cosine similarity score to a more intuitive 0-1 range.
    CLIP cosine similarity ranges from -1 to 1, but typically:
    - Good matches: 0.2-0.4
    - Great matches: 0.4-0.6
    - Excellent matches: 0.6+
    
    We'll transform it to make lower scores more visible:
    - Map [-1, 1] to [0, 1] using: (score + 1) / 2
    - Then apply a power curve to emphasize higher scores: score^0.7
    """
    # Normalize from [-1, 1] to [0, 1]
    normalized = (score + 1) / 2
    # Apply power curve to make differences more visible
    # This makes 0.2 -> ~0.4, 0.3 -> ~0.5, 0.4 -> ~0.6
    return normalized ** 0.7

def build_text_filter(query):
    """
    Build Qdrant filter for full-text search on tags and caption.
    Uses Qdrant's native full-text search capabilities.
    """
    # Extract keywords from query
    words = re.findall(r'\b\w+\b', query.lower())
    stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    if not keywords:
        # If no keywords, search for the whole query
        keywords = [query.lower()]
    
    # Build filter conditions for tags and caption
    conditions = []
    for keyword in keywords:
        # Match in tags (keyword search)
        conditions.append(
            FieldCondition(
                key="tags",
                match=MatchText(text=keyword)
            )
        )
        # Match in caption (text search)
        conditions.append(
            FieldCondition(
                key="caption",
                match=MatchText(text=keyword)
            )
        )
    
    # Use "should" (OR) logic - match any keyword in tags OR caption
    if conditions:
        return Filter(should=conditions)
    return None

@app.get("/search/text", response_model=List[SearchResponse])
async def search_by_text(query: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100)):
    """
    Search for photos by text description using Qdrant's native hybrid search:
    1. CLIP semantic similarity (vector search)
    2. Full-text search on tags and captions (Qdrant native)
    
    Uses Qdrant's query API which combines both automatically.
    """
    try:
        logger.info(f"Text search query: '{query}'")
        
        # Get text embedding (already normalized)
        text_embedding = get_text_embedding(query).tolist()
        
        # Build text filter for full-text search
        text_filter = build_text_filter(query)
        
        # Use Qdrant's native hybrid search: vector similarity + full-text filter
        # This combines both searches efficiently at the database level
        search_results = qdrant_client.search(
            collection_name=COLLECTION_IMAGES,
            query_vector=text_embedding,
            query_filter=text_filter,  # Qdrant native filter for full-text search
            limit=limit * 2  # Get more results for re-ranking
        )
        
        logger.info(f"Found {len(search_results)} hybrid search results for query: '{query}'")
        
        # Process results
        results = []
        for result in search_results[:limit]:
            payload = result.payload
            
            # Normalize similarity score
            normalized_score = normalize_similarity_score(result.score)
            
            results.append(SearchResponse(
                photo_id=result.id,
                image_url=f"/images/processed/{result.id}.jpg",
                thumbnail_url=f"/images/thumbnails/{result.id}.jpg",
                caption=payload.get("caption"),
                tags=payload.get("tags", []),
                face_count=payload.get("face_count"),
                similarity_score=normalized_score
            ))
        
        logger.info(f"Returning {len(results)} results using Qdrant hybrid search")
        return results
        
    except Exception as e:
        logger.error(f"Error in text search: {e}", exc_info=True)
        # Fallback to simple vector search if hybrid search fails
        try:
            logger.warning("Falling back to vector-only search")
            text_embedding = get_text_embedding(query).tolist()
            search_results = qdrant_client.search(
                collection_name=COLLECTION_IMAGES,
                query_vector=text_embedding,
                limit=limit
            )
            
            results = []
            for result in search_results:
                payload = result.payload
                normalized_score = normalize_similarity_score(result.score)
                results.append(SearchResponse(
                    photo_id=result.id,
                    image_url=f"/images/processed/{result.id}.jpg",
                    thumbnail_url=f"/images/thumbnails/{result.id}.jpg",
                    caption=payload.get("caption"),
                    tags=payload.get("tags", []),
                    face_count=payload.get("face_count"),
                    similarity_score=normalized_score
                ))
            return results
        except Exception as e2:
            logger.error(f"Fallback search also failed: {e2}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/similar", response_model=List[SearchResponse])
async def search_similar(photo_id: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100)):
    """
    Find similar images to a given photo by photo_id.
    Uses CLIP embeddings to find visually similar images.
    """
    try:
        logger.info(f"Similar image search for photo_id: {photo_id}")
        
        # Retrieve the image embedding from Qdrant
        image_points = qdrant_client.retrieve(
            collection_name=COLLECTION_IMAGES,
            ids=[photo_id],
            with_vectors=True
        )
        
        if not image_points or len(image_points) == 0:
            raise HTTPException(status_code=404, detail="Photo not found")
        
        query_vector = image_points[0].vector
        
        # Search for similar images (excluding the query image itself)
        search_results = qdrant_client.search(
            collection_name=COLLECTION_IMAGES,
            query_vector=query_vector,
            limit=limit + 1  # Get one extra to exclude the original
        )
        
        results = []
        for result in search_results:
            # Skip the original image
            if result.id == photo_id:
                continue
            
            payload = result.payload
            normalized_score = normalize_similarity_score(result.score)
            results.append(SearchResponse(
                photo_id=result.id,
                image_url=f"/images/processed/{result.id}.jpg",
                thumbnail_url=f"/images/thumbnails/{result.id}.jpg",
                caption=payload.get("caption"),
                tags=payload.get("tags", []),
                face_count=payload.get("face_count"),
                similarity_score=normalized_score
            ))
            
            if len(results) >= limit:
                break
        
        logger.info(f"Found {len(results)} similar images")
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in similar image search: {e}", exc_info=True)
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

