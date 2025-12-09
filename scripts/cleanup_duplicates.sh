#!/bin/bash
# Script to clean up duplicate directories in backend/

echo "=========================================="
echo "Cleaning Up Duplicate Directories"
echo "=========================================="
echo ""

BACKEND_DIR="$(cd "$(dirname "$0")/.." && pwd)/backend"

echo "Checking for duplicate directories in backend/..."
echo ""

# Check and remove duplicate directories
if [ -d "$BACKEND_DIR/images" ]; then
    echo "Found: backend/images/"
    echo "  This should be removed (using root ./images/ instead)"
    read -p "  Remove backend/images/? (y/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$BACKEND_DIR/images"
        echo "  ✓ Removed backend/images/"
    fi
fi

if [ -d "$BACKEND_DIR/models" ]; then
    echo "Found: backend/models/"
    echo "  This should be removed (using root ./models/ instead)"
    read -p "  Remove backend/models/? (y/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$BACKEND_DIR/models"
        echo "  ✓ Removed backend/models/"
    fi
fi

if [ -d "$BACKEND_DIR/qdrant_storage" ]; then
    echo "Found: backend/qdrant_storage/"
    echo "  This should be removed (using root ./qdrant_storage/ instead)"
    read -p "  Remove backend/qdrant_storage/? (y/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$BACKEND_DIR/qdrant_storage"
        echo "  ✓ Removed backend/qdrant_storage/"
    fi
fi

echo ""
echo "=========================================="
echo "Cleanup Complete"
echo "=========================================="
echo ""
echo "All data should now be in root directories:"
echo "  - ./images/ (raw, processed, thumbnails, processing)"
echo "  - ./models/ (insightface, sentence_transformers, huggingface)"
echo "  - ./qdrant_storage/ (database files)"
echo ""

