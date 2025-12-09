"""
Image utility functions.
"""
import os
import cv2
import numpy as np
from PIL import Image
from typing import Union

# Register pillow-heif plugin for HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass
except Exception:
    pass

def load_image_as_cv2(image_path: Union[str, np.ndarray]) -> np.ndarray:
    """
    Load an image as OpenCV format (BGR numpy array).
    Handles HEIC files by converting through PIL first.
    
    Args:
        image_path: Path to image or numpy array
    
    Returns:
        Image as BGR numpy array
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

def load_image_as_pil(image_path: Union[str, Image.Image]) -> Image.Image:
    """
    Load an image as PIL Image.
    Handles HEIC files.
    
    Args:
        image_path: Path to image or PIL Image
    
    Returns:
        PIL Image in RGB mode
    """
    if isinstance(image_path, str):
        img = Image.open(image_path)
    else:
        img = image_path
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return img

def create_thumbnail(
    image_path: str, 
    output_path: str, 
    max_size: int = 300,
    quality: int = 85
) -> bool:
    """
    Create a thumbnail of an image.
    
    Args:
        image_path: Path to source image
        output_path: Path to save thumbnail
        max_size: Maximum size for thumbnail
        quality: JPEG quality (1-100)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        img.save(output_path, "JPEG", quality=quality)
        return True
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error creating thumbnail: {e}")
        return False

def convert_heic_to_jpg(image_path: str, output_path: str, quality: int = 95) -> bool:
    """
    Convert HEIC image to JPG.
    
    Args:
        image_path: Path to HEIC image
        output_path: Path to save JPG
        quality: JPEG quality (1-100)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(output_path, "JPEG", quality=quality)
        return True
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error converting HEIC to JPG: {e}")
        return False

