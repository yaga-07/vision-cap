# Vision Cap - Local Event Photo Search System

A privacy-first, offline-capable photo search system for events. Guests can find their photos using face recognition or text search, all running locally without internet.

## Features

- 🔍 **Face Search**: Upload a selfie to find all photos containing you
- 📝 **Text Search**: Search photos by description (e.g., "dancing", "red dress")
- 📸 **Live Feed**: View all processed photos in real-time
- 🔒 **100% Offline**: Everything runs locally, no cloud required
- ⚡ **Fast Processing**: Images processed in <3 seconds
- 📱 **Mobile-First UI**: Beautiful, responsive web interface

## Architecture

- **Qdrant**: Vector database for face and image embeddings
- **InsightFace**: Face detection and recognition
- **CLIP**: Semantic image search
- **FastAPI**: REST API backend
- **React**: Modern web frontend

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM recommended
- GPU optional (CPU works fine)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd vision_cap
   ```

2. **Create necessary directories**
   ```bash
   mkdir -p images/raw images/processed images/thumbnails images/processing
   mkdir -p models
   ```
   
   Note: Models will be automatically downloaded on first run (~1-2GB). They are stored in the `models/` directory.

3. **Start the services**
   ```bash
   docker-compose up --build
   ```

   This will start:
   - Qdrant on port 6333
   - Backend API on port 8000
   - Frontend on port 3000

4. **Start the worker** (in a separate terminal)
   ```bash
   docker-compose exec backend python worker.py
   ```

5. **Add photos for processing**
   ```bash
   # Copy photos to the raw directory
   cp your-photos/*.jpg images/raw/
   ```

6. **Access the application**
   - Frontend: http://localhost:3000
   - API Docs: http://localhost:8000/docs
   - Qdrant Dashboard: http://localhost:6333/dashboard

## Usage

### For Guests

1. Open http://localhost:3000 in your browser
2. Choose search mode:
   - **Text Search**: Type a description (e.g., "dancing", "cake")
   - **Find Me**: Click "Find Me" and take a selfie
3. Browse results and download photos

### For Photographers

1. Switch to "Photographer" view in the app
2. View statistics:
   - Images processed
   - Unique faces detected
   - System status
3. Monitor the live feed of processed photos
4. Share the URL with guests (configure mDNS for `photos.local`)

## API Endpoints

- `POST /search/face` - Search by face image
- `POST /search/text` - Search by text query
- `GET /feed` - Get paginated photo feed
- `GET /stats` - Get system statistics
- `GET /health` - Health check

See http://localhost:8000/docs for interactive API documentation.

## Development

### Backend Development

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Development

```bash
cd frontend
npm install
npm run dev
```

### Running the Worker

```bash
cd backend
python worker.py
```

The worker watches `images/raw/` for new images and processes them automatically.

## Configuration

### Environment Variables

**Backend:**
- `QDRANT_HOST`: Qdrant host (default: localhost)
- `QDRANT_PORT`: Qdrant port (default: 6333)

**Frontend:**
- `VITE_API_URL`: Backend API URL (default: http://localhost:8000)

### Directory Structure

```
vision_cap/
├── backend/          # FastAPI backend
├── frontend/         # React frontend
├── images/           # Photo storage
│   ├── raw/          # Drop photos here for processing
│   ├── processed/    # Processed full-size images
│   └── thumbnails/   # Generated thumbnails
├── qdrant_storage/   # Qdrant database files
└── docker-compose.yml
```

## Troubleshooting

### Models not loading
- Ensure you have enough RAM (8GB+ recommended)
- First run downloads models (~2GB), be patient
- Check Docker logs: `docker-compose logs backend`

### Photos not processing
- Check worker is running: `docker-compose exec backend python worker.py`
- Verify images are in `images/raw/` directory
- Check file permissions

### Face search not working
- Ensure photo has clear faces (40x40px minimum)
- Try adjusting similarity threshold in API call
- Check Qdrant is running: `docker-compose ps`

## Performance Tips

- **GPU**: For faster processing, ensure GPU access in Docker
- **Batch Processing**: Process photos in batches for better throughput
- **Storage**: Use SSD for image storage for faster I/O

## Model Management

Models are stored locally in the `models/` directory (~1-2GB total):
- `models/insightface/` - Face recognition models (~500MB)
- `models/sentence_transformers/` - CLIP models (~600MB)
- `models/huggingface/` - HuggingFace transformers (if used)

### Downloading Models

Models are automatically downloaded on first startup. To download manually:

```bash
# Method 1: Using Docker (Recommended - no local dependencies needed)
./scripts/download_models_docker.sh

# Method 2: Using Docker Compose directly
docker-compose exec backend python scripts/download_models.py

# Method 3: Local Python (requires dependencies installed)
pip install -r backend/requirements.txt
python scripts/download_models.py
```

**Note**: If running locally, ensure you have all dependencies installed. The Docker method is recommended as it handles all dependencies automatically.

### Clearing Models

To free up disk space, you can clear downloaded models:

```bash
# Using the clear script
./scripts/clear_models.sh

# Or manually
rm -rf models/insightface/* models/sentence_transformers/* models/huggingface/*
```

Models will be automatically re-downloaded on next startup if missing.

## Future Enhancements

- [ ] Florence-2 integration for detailed captions
- [ ] mDNS/Bonjour configuration for `photos.local`
- [ ] PWA support for offline mobile app
- [ ] Event management (create multiple events)
- [ ] QR code generation for easy sharing
- [ ] Advanced filtering and sorting

## License

MIT License

## Support

For issues and questions, please open an issue on GitHub.

