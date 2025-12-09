# Project Structure

```
vision_cap/
├── backend/                    # FastAPI Backend
│   ├── Dockerfile             # Backend container definition
│   ├── main.py                # FastAPI application & API endpoints
│   ├── models.py              # AI model loading & inference
│   ├── worker.py              # Image processing worker (watchdog)
│   └── requirements.txt       # Python dependencies
│
├── frontend/                   # React Frontend
│   ├── Dockerfile             # Frontend container definition
│   ├── index.html             # HTML entry point
│   ├── package.json           # Node dependencies
│   ├── vite.config.js         # Vite configuration
│   └── src/
│       ├── main.jsx           # React entry point
│       ├── App.jsx            # Main app component
│       ├── App.css            # App styles
│       ├── index.css          # Global styles
│       └── components/
│           ├── GuestView.jsx   # Guest search interface
│           ├── GuestView.css
│           ├── PhotographerView.jsx  # Photographer dashboard
│           └── PhotographerView.css
│
├── docker-compose.yml         # Main Docker Compose config
├── docker-compose.worker.yml  # Worker-only config (optional)
├── start.sh                   # Startup script
├── README.md                  # Main documentation
├── QUICKSTART.md             # Quick start guide
└── buildplan.md              # Original build plan

# Runtime directories (created on first run)
├── images/                    # Image storage
│   ├── raw/                   # Drop photos here
│   ├── processing/            # Temporary processing queue
│   ├── processed/             # Processed full-size images
│   └── thumbnails/            # Generated thumbnails
└── qdrant_storage/           # Qdrant database files
```

## Key Components

### Backend (`backend/`)
- **main.py**: FastAPI server with REST endpoints
  - `POST /search/face` - Face-based search
  - `GET /search/text` - Text-based search
  - `GET /feed` - Paginated photo feed
  - `GET /stats` - System statistics
  
- **models.py**: AI model management
  - InsightFace (face detection/recognition)
  - CLIP (semantic search)
  - Model loading and caching

- **worker.py**: Background image processor
  - Watches `images/raw/` directory
  - Processes images automatically
  - Extracts faces, embeddings, metadata
  - Stores in Qdrant

### Frontend (`frontend/`)
- **GuestView**: Search interface for guests
  - Text search input
  - Camera-based face search
  - Results grid with download
  
- **PhotographerView**: Dashboard for photographers
  - Statistics display
  - Live feed of processed photos
  - System status

### Docker Services
1. **qdrant**: Vector database (port 6333)
2. **backend**: FastAPI server (port 8000)
3. **frontend**: React app (port 3000)
4. **worker**: Image processing service

## Data Flow

1. **Photo Ingestion**:
   ```
   images/raw/*.jpg → worker.py → Qdrant
   ```

2. **Face Search**:
   ```
   Guest uploads selfie → Backend → InsightFace → Qdrant → Results
   ```

3. **Text Search**:
   ```
   Guest types query → Backend → CLIP → Qdrant → Results
   ```

4. **Photo Serving**:
   ```
   Frontend requests → Backend static files → images/processed/
   ```

## Environment Variables

### Backend
- `QDRANT_HOST`: Qdrant hostname (default: localhost)
- `QDRANT_PORT`: Qdrant port (default: 6333)

### Frontend
- `VITE_API_URL`: Backend API URL (default: http://localhost:8000)

## Dependencies

### Backend (Python)
- FastAPI, Uvicorn (web framework)
- InsightFace (face recognition)
- Sentence Transformers (CLIP)
- Qdrant Client (vector DB)
- OpenCV, Pillow (image processing)
- Watchdog (file monitoring)

### Frontend (Node.js)
- React 18
- Vite (build tool)
- Axios (HTTP client)
- React Webcam (camera access)

