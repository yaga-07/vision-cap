#!/bin/bash
# Script to clear downloaded model weights to free up space

MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"

echo "=========================================="
echo "Clear Model Weights"
echo "=========================================="
echo ""
echo "This will delete all downloaded model weights from:"
echo "$MODELS_DIR"
echo ""
echo "Models will be re-downloaded automatically on next startup."
echo ""
read -p "Are you sure you want to continue? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Clearing models..."
    
    # Remove model files but keep directory structure
    rm -rf "$MODELS_DIR/insightface"/*
    rm -rf "$MODELS_DIR/sentence_transformers"/*
    rm -rf "$MODELS_DIR/huggingface"/*
    
    echo "✓ Models cleared successfully"
    echo ""
    echo "Disk space freed: ~1-2GB"
    echo ""
    echo "Models will be automatically re-downloaded on next startup."
else
    echo "Cancelled."
fi

