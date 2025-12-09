#!/usr/bin/env python3
"""
Script to download and cache all required models locally.
This ensures models are stored in the models/ directory.

Requirements:
    - Install dependencies: pip install -r backend/requirements.txt
    - Or use Docker: docker-compose exec backend python scripts/download_models.py
"""
import os
import sys
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set model directories
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
INSIGHTFACE_DIR = os.path.join(MODELS_DIR, 'insightface')
SENTENCE_TRANSFORMERS_DIR = os.path.join(MODELS_DIR, 'sentence_transformers')
HUGGINGFACE_DIR = os.path.join(MODELS_DIR, 'huggingface')

# Create directories
os.makedirs(INSIGHTFACE_DIR, exist_ok=True)
os.makedirs(SENTENCE_TRANSFORMERS_DIR, exist_ok=True)
os.makedirs(HUGGINGFACE_DIR, exist_ok=True)

# Set environment variables
os.environ["INSIGHTFACE_ROOT"] = INSIGHTFACE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = SENTENCE_TRANSFORMERS_DIR
os.environ["HF_HOME"] = HUGGINGFACE_DIR
os.environ["TRANSFORMERS_CACHE"] = HUGGINGFACE_DIR

def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    try:
        import insightface
    except ImportError:
        missing.append("insightface")
    
    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime")
    
    try:
        import sentence_transformers
    except ImportError:
        missing.append("sentence-transformers")
    
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import timm
    except ImportError:
        missing.append("timm")
    
    try:
        import einops
    except ImportError:
        missing.append("einops")
    
    if missing:
        print("=" * 60)
        print("Missing Dependencies")
        print("=" * 60)
        print(f"The following packages are required but not installed: {', '.join(missing)}")
        print()
        print("To install dependencies:")
        print("  1. Install locally:")
        print("     pip install -r backend/requirements.txt")
        print()
        print("  2. Or use Docker (recommended):")
        print("     docker-compose up -d backend")
        print("     docker-compose exec backend python scripts/download_models.py")
        print()
        print("  3. Or install minimal dependencies:")
        print(f"     pip install {' '.join(missing)}")
        print()
        return False
    return True

print("=" * 60)
print("Downloading AI Models for Vision Cap")
print("=" * 60)
print(f"Models will be saved to: {MODELS_DIR}")
print()

# Check dependencies
if not check_dependencies():
    sys.exit(1)

# Download InsightFace
print("1. Downloading InsightFace (buffalo_l)...")
try:
    from insightface.app import FaceAnalysis
    face_app = FaceAnalysis(name='buffalo_l', root=INSIGHTFACE_DIR)
    face_app.prepare(ctx_id=-1)
    print("   ✓ InsightFace downloaded successfully")
    print(f"   Location: {INSIGHTFACE_DIR}")
except ImportError as e:
    print(f"   ✗ Missing dependency: {e}")
    print()
    print("   Please install dependencies:")
    print("     pip install -r backend/requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"   ✗ Error downloading InsightFace: {e}")
    sys.exit(1)

print()

# Download CLIP
print("2. Downloading CLIP (clip-ViT-B-32)...")
try:
    from sentence_transformers import SentenceTransformer
    clip_model = SentenceTransformer(
        'clip-ViT-B-32',
        cache_folder=SENTENCE_TRANSFORMERS_DIR
    )
    print("   ✓ CLIP downloaded successfully")
    print(f"   Location: {SENTENCE_TRANSFORMERS_DIR}")
except Exception as e:
    print(f"   ✗ Error downloading CLIP: {e}")
    sys.exit(1)

print()

# Download Florence-2-base
print("3. Downloading Florence-2-base (for tag and caption extraction)...")
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    import torch
    
    print("   Downloading processor...")
    florence_processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base",
        cache_dir=HUGGINGFACE_DIR,
        trust_remote_code=True
    )
    print("   ✓ Processor downloaded")
    
    print("   Downloading model (this may take a while, ~1.5GB)...")
    florence_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base",
        cache_dir=HUGGINGFACE_DIR,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    print("   ✓ Florence-2-base downloaded successfully")
    print(f"   Location: {HUGGINGFACE_DIR}")
    print("   Note: Model will be loaded into memory when needed")
except ImportError as e:
    print(f"   ✗ Missing dependency: {e}")
    print("   Please install required dependencies:")
    print("     pip install transformers torch timm einops")
    print("   Or install all requirements:")
    print("     pip install -r backend/requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"   ✗ Error downloading Florence-2-base: {e}")
    print("   Warning: Tag and caption extraction will be disabled")
    print("   You can continue without Florence-2, but text search accuracy may be reduced")
    import traceback
    traceback.print_exc()
    # Don't exit - allow the script to continue
    print("   Continuing without Florence-2...")

print()
print("=" * 60)
print("Model Download Complete!")
print("=" * 60)
print()
print("Model sizes:")
print(f"  InsightFace: ~500MB")
print(f"  CLIP: ~600MB")
print(f"  Florence-2-base: ~1.5GB")
print(f"  Total: ~2.6GB")
print()
print("Models are now cached locally and will be reused on next startup.")
print()
print("Note: Florence-2-base is optional but recommended for better text search accuracy.")
print("      It extracts object tags and detailed captions from images.")
print()

