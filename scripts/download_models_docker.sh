#!/bin/bash
# Download models using Docker (recommended method)
# This ensures all dependencies are available

echo "=========================================="
echo "Downloading Models via Docker"
echo "=========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if backend container exists
if ! docker-compose ps backend | grep -q "Up"; then
    echo "Starting backend container..."
    docker-compose up -d backend
    sleep 5
fi

echo "Downloading models..."
echo ""

# Run the download script inside the container
docker-compose exec backend python scripts/download_models.py

echo ""
echo "=========================================="
echo "Models downloaded successfully!"
echo "=========================================="
echo ""
echo "Models are stored in: ./models/"
echo "You can now clear them anytime with: ./scripts/clear_models.sh"
echo ""

