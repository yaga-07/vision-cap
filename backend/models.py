"""
Model loading and inference utilities.
Loads models once and keeps them in memory for fast inference.
Models are stored locally in /app/models/ directory.
"""
import os
import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

# Model storage paths
MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")
INSIGHTFACE_DIR = os.path.join(MODELS_DIR, "insightface")
SENTENCE_TRANSFORMERS_DIR = os.path.join(MODELS_DIR, "sentence_transformers")
HUGGINGFACE_DIR = os.path.join(MODELS_DIR, "huggingface")

# Ensure model directories exist
os.makedirs(INSIGHTFACE_DIR, exist_ok=True)
os.makedirs(SENTENCE_TRANSFORMERS_DIR, exist_ok=True)
os.makedirs(HUGGINGFACE_DIR, exist_ok=True)

# Set environment variables for model storage
os.environ["INSIGHTFACE_ROOT"] = INSIGHTFACE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = SENTENCE_TRANSFORMERS_DIR
os.environ["HF_HOME"] = HUGGINGFACE_DIR
os.environ["TRANSFORMERS_CACHE"] = HUGGINGFACE_DIR

# Global model instances
face_app = None
clip_model = None
florence_processor = None
florence_model = None

def load_models():
    """
    Load all AI models once at startup.
    Models are stored in /app/models/ directory.
    Returns: (face_app, clip_model)
    """
    global face_app, clip_model, florence_processor, florence_model
    
    print("Loading models...")
    print(f"Models directory: {MODELS_DIR}")
    
    # Load InsightFace for face detection and embedding
    try:
        # InsightFace will download to root directory if not present
        # Set root to our models directory
        face_app = FaceAnalysis(
            name='buffalo_l',
            root=INSIGHTFACE_DIR,
            providers=['CPUExecutionProvider']
        )
        face_app.prepare(ctx_id=-1)  # -1 for CPU, 0 for GPU
        print(f"✓ InsightFace loaded from {INSIGHTFACE_DIR}")
    except Exception as e:
        print(f"Error loading InsightFace: {e}")
        # Fallback: try without explicit providers
        try:
            face_app = FaceAnalysis(name='buffalo_l', root=INSIGHTFACE_DIR)
            face_app.prepare(ctx_id=-1)
            print(f"✓ InsightFace loaded (fallback) from {INSIGHTFACE_DIR}")
        except Exception as e2:
            print(f"Failed to load InsightFace: {e2}")
            raise
    
    # Load CLIP for semantic search
    try:
        # SentenceTransformer will use SENTENCE_TRANSFORMERS_HOME
        clip_model = SentenceTransformer(
            'clip-ViT-B-32',
            cache_folder=SENTENCE_TRANSFORMERS_DIR
        )
        print(f"✓ CLIP loaded from {SENTENCE_TRANSFORMERS_DIR}")
    except Exception as e:
        print(f"Error loading CLIP: {e}")
        raise
    
    # Load Florence-2 for captioning (optional, can be slow)
    # try:
    #     florence_processor = AutoProcessor.from_pretrained(
    #         "microsoft/Florence-2-large",
    #         cache_dir=HUGGINGFACE_DIR,
    #         trust_remote_code=True
    #     )
    #     florence_model = AutoModelForCausalLM.from_pretrained(
    #         "microsoft/Florence-2-large",
    #         cache_dir=HUGGINGFACE_DIR,
    #         trust_remote_code=True
    #     )
    #     print(f"✓ Florence-2 loaded from {HUGGINGFACE_DIR}")
    # except Exception as e:
    #     print(f"Warning: Florence-2 not loaded: {e}")
    
    print("All models loaded successfully!")
    return face_app, clip_model

def get_face_embedding(image_path_or_array):
    """
    Extract face embeddings from an image.
    Args:
        image_path_or_array: Path to image or numpy array
    Returns:
        List of face embeddings (each is 512-dim numpy array)
    """
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
    else:
        img = image_path_or_array
    
    if img is None:
        return []
    
    faces = face_app.get(img)
    embeddings = []
    
    for face in faces:
        # Only process faces larger than 40x40 pixels
        bbox = face.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        if width >= 40 and height >= 40:
            embeddings.append(face.embedding)
    
    return embeddings

def get_clip_embedding(image_path_or_pil):
    """
    Get CLIP embedding for an image.
    Args:
        image_path_or_pil: Path to image or PIL Image
    Returns:
        512-dim numpy array
    """
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil)
    else:
        img = image_path_or_pil
    
    embedding = clip_model.encode(img)
    return embedding

def get_text_embedding(text):
    """
    Get CLIP embedding for text.
    """
    return clip_model.encode([text])[0]

def get_florence_caption(image_path_or_pil):
    """
    Get detailed caption from Florence-2.
    Note: This is slow and optional for MVP.
    """
    if florence_model is None or florence_processor is None:
        return None
    
    try:
        if isinstance(image_path_or_pil, str):
            img = Image.open(image_path_or_pil)
        else:
            img = image_path_or_pil
        
        prompt = "<DETAILED_CAPTION>"
        inputs = florence_processor(text=prompt, images=img, return_tensors="pt")
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=50,
            num_beams=3,
        )
        generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = florence_processor.post_process_generate(generated_text, prompt=prompt)
        
        return parsed_answer if parsed_answer else None
    except Exception as e:
        print(f"Error getting Florence caption: {e}")
        return None
