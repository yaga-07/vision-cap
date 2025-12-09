#!/bin/bash

# Vision Cap Startup Script

echo "🚀 Starting Vision Cap..."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p images/raw images/processed images/thumbnails images/processing
mkdir -p qdrant_storage

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose up -d --build

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
echo "🔍 Checking service status..."
docker-compose ps

echo ""
echo "✅ Vision Cap is starting!"
echo ""
echo "📸 Frontend: http://localhost:3000"
echo "🔌 API: http://localhost:8000"
echo "📊 API Docs: http://localhost:8000/docs"
echo "🗄️  Qdrant: http://localhost:6333"
echo ""
echo "To start the worker (in a separate terminal):"
echo "  docker-compose exec backend python worker.py"
echo ""
echo "To add photos for processing:"
echo "  cp your-photos/*.jpg images/raw/"
echo ""

