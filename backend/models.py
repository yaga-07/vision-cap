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

# Register pillow-heif plugin for HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("Warning: pillow-heif not available. HEIC support will be limited.")
except Exception as e:
    print(f"Warning: Could not register HEIC opener: {e}")

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
    Returns: (face_app, clip_model, florence_processor, florence_model)
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
    
    # Load Florence-2-base for captioning and object detection (smaller and faster than large)
    try:
        florence_processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base",
            cache_dir=HUGGINGFACE_DIR,
            trust_remote_code=True
        )
        
        # Determine device: MPS (Mac GPU) > CUDA > CPU
        if torch.backends.mps.is_available():
            device = "mps"
            torch_dtype = torch.float32  # MPS supports float32
            print("Using MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16  # CUDA supports float16
            print("Using CUDA (NVIDIA GPU)")
        else:
            device = "cpu"
            torch_dtype = torch.float32  # CPU uses float32
            print("Using CPU")
        
        florence_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base",
            cache_dir=HUGGINGFACE_DIR,
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )
        
        # Move model to appropriate device
        florence_model = florence_model.to(device)
        florence_model.eval()  # Set to evaluation mode
        print(f"✓ Florence-2-base loaded from {HUGGINGFACE_DIR} on {device}")
    except Exception as e:
        print(f"Warning: Florence-2-base not loaded: {e}")
        print("Tag and caption extraction will be limited. Continuing without Florence-2...")
        import traceback
        traceback.print_exc()
        florence_processor = None
        florence_model = None
    
    print("All models loaded successfully!")
    return face_app, clip_model, florence_processor, florence_model

def load_image_as_cv2(image_path):
    """
    Load an image as OpenCV format (BGR numpy array).
    Handles HEIC files by converting through PIL first.
    """
    if not isinstance(image_path, str):
        return image_path
    
    # Check if it's a HEIC file
    ext = os.path.splitext(image_path)[1].lower()
    if ext in ['.heic', '.heif']:
        # Load via PIL first (pillow-heif handles HEIC)
        pil_img = Image.open(image_path)
        # Convert PIL RGB to OpenCV BGR
        img_array = np.array(pil_img.convert('RGB'))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_bgr
    else:
        # Use OpenCV directly for other formats
        return cv2.imread(image_path)

def get_face_embedding(image_path_or_array):
    """
    Extract face embeddings from an image.
    Args:
        image_path_or_array: Path to image or numpy array
    Returns:
        List of face embeddings (each is 512-dim numpy array)
    """
    if isinstance(image_path_or_array, str):
        img = load_image_as_cv2(image_path_or_array)
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
        512-dim numpy array (normalized)
    """
    if isinstance(image_path_or_pil, str):
        # PIL with pillow-heif can handle HEIC directly
        img = Image.open(image_path_or_pil)
        # Convert to RGB if needed (HEIC might be RGBA)
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        img = image_path_or_pil
        # Ensure RGB mode
        if hasattr(img, 'mode') and img.mode != 'RGB':
            img = img.convert('RGB')
    
    embedding = clip_model.encode(img, normalize_embeddings=True)
    return embedding

def get_text_embedding(text):
    """
    Get CLIP embedding for text.
    Returns normalized embedding.
    """
    return clip_model.encode([text], normalize_embeddings=True)[0]

def get_florence_tags_and_caption(image_path_or_pil):
    """
    Extract object detection tags and detailed caption from Florence-2.
    Returns: (tags_list, caption_string)
    """
    if florence_model is None or florence_processor is None:
        return [], None
    
    try:
        if isinstance(image_path_or_pil, str):
            img = Image.open(image_path_or_pil)
            if img.mode != 'RGB':
                img = img.convert('RGB')
        else:
            img = image_path_or_pil
            if hasattr(img, 'mode') and img.mode != 'RGB':
                img = img.convert('RGB')
        
        # Determine device from model
        device = next(florence_model.parameters()).device
        
        tags = []
        caption = None
        
        # Task 1: Object Detection
        try:
            od_prompt = "<OD>"
            od_inputs = florence_processor(text=od_prompt, images=img, return_tensors="pt")
            # Move inputs to same device as model
            od_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in od_inputs.items()}
            
            with torch.no_grad():
                od_generated_ids = florence_model.generate(
                    input_ids=od_inputs["input_ids"],
                    pixel_values=od_inputs["pixel_values"],
                    max_new_tokens=100,
                    num_beams=3,
                    do_sample=False
                )
            
            od_generated_text = florence_processor.batch_decode(od_generated_ids, skip_special_tokens=False)[0]
            od_parsed = florence_processor.post_process_generate(od_generated_text, prompt=od_prompt)
            
            if od_parsed:
                # Parse object detection results
                # Format is typically: "object1. object2. object3."
                od_text = str(od_parsed).strip()
                if od_text:
                    # Split by periods and clean up
                    tags = [tag.strip().lower() for tag in od_text.split('.') if tag.strip()]
                    # Remove duplicates while preserving order
                    tags = list(dict.fromkeys(tags))
        except Exception as e:
            print(f"Error in object detection: {e}")
        
        # Task 2: Detailed Caption
        try:
            caption_prompt = "<DETAILED_CAPTION>"
            caption_inputs = florence_processor(text=caption_prompt, images=img, return_tensors="pt")
            # Move inputs to same device as model
            caption_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in caption_inputs.items()}
            
            with torch.no_grad():
                caption_generated_ids = florence_model.generate(
                    input_ids=caption_inputs["input_ids"],
                    pixel_values=caption_inputs["pixel_values"],
                    max_new_tokens=100,
                    num_beams=3,
                    do_sample=False
                )
            
            caption_generated_text = florence_processor.batch_decode(caption_generated_ids, skip_special_tokens=False)[0]
            # caption_parsed = florence_processor.post_process_generate(caption_generated_text, prompt=caption_prompt)
            
            if caption_generated_text:
                caption = str(caption_generated_text).strip()
        except Exception as e:
            print(f"Error in caption generation: {e}")
        
        return tags, caption
        
    except Exception as e:
        print(f"Error getting Florence tags and caption: {e}")
        return [], None
