# Quick Start Guide

## Prerequisites
- Docker Desktop installed and running
- At least 8GB RAM available
- 10GB free disk space (for models and images)

## Step-by-Step Setup

### 1. Start All Services
```bash
docker-compose up --build
```

This will:
- Download and start Qdrant (vector database)
- Build and start the FastAPI backend
- Build and start the React frontend
- Start the image processing worker

**Note:** First startup takes 5-10 minutes as it downloads AI models (~2GB).

### 2. Verify Services Are Running
```bash
docker-compose ps
```

You should see:
- `qdrant` - Running
- `backend` - Running  
- `frontend` - Running
- `worker` - Running

### 3. Add Photos for Processing
```bash
# Copy your photos to the raw directory
cp /path/to/your/photos/*.jpg images/raw/
```

The worker will automatically:
- Detect faces
- Extract embeddings
- Process images
- Store in database

### 4. Access the Application

- **Frontend (Guest/Photographer UI)**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## Testing the System

### Test Face Search
1. Go to http://localhost:3000
2. Click "Find Me" tab
3. Allow camera access
4. Take a selfie
5. See matching photos!

### Test Text Search
1. Go to http://localhost:3000
2. Type a search query like "dancing" or "cake"
3. Click Search
4. Browse results

### Check Photographer Dashboard
1. Go to http://localhost:3000
2. Click "Photographer" tab
3. View statistics and live feed

## Troubleshooting

### Services won't start
```bash
# Check logs
docker-compose logs

# Restart services
docker-compose down
docker-compose up --build
```

### Worker not processing images
```bash
# Check worker logs
docker-compose logs worker

# Verify images directory exists
ls -la images/raw/
```

### Models not loading
- First run downloads models (~2GB)
- Check internet connection
- Wait 5-10 minutes for download

### Port conflicts
If ports 3000, 8000, or 6333 are in use:
```bash
# Edit docker-compose.yml and change port mappings
# Example: "3001:3000" instead of "3000:3000"
```

## Stopping the System
```bash
docker-compose down
```

To also remove volumes (deletes all data):
```bash
docker-compose down -v
```

## Next Steps
- Configure mDNS for `photos.local` access
- Set up Wi-Fi hotspot for guests
- Add more photos and test search accuracy
- Customize UI colors/branding

