"""
Application settings and configuration.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    
    # Model Directories
    models_dir: str = "./models"
    log_dir: str = "./logs"
    
    # Image Directories
    raw_dir: str = "./images/raw"
    processing_dir: str = "./images/processing"
    processed_dir: str = "./images/processed"
    thumbnail_dir: str = "./images/thumbnails"
    
    # Collections
    collection_images: str = "images"
    collection_faces: str = "faces"
    
    # Model Configuration
    face_detection_model: str = "insightface"  # Options: insightface
    face_detection_model_name: str = "buffalo_l"
    
    embedding_model: str = "clip"  # Options: clip, openai-clip, etc.
    embedding_model_name: str = "clip-ViT-B-32"
    
    vlm_model: str = "google-genai"  # Options: florence2, genai, google-genai, etc.
    # vlm_model_name: str = "microsoft/Florence-2-base"
    
    # Google GenAI Configuration
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    google_genai_model: str = "gemini-flash-lite-latest"  # Options: gemini-1.5-flash, gemini-2.5-pro
    
    # Vector Configuration
    vector_size: int = 512
    face_min_size: int = 40  # Minimum face size in pixels
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Logging
    log_level: str = "INFO"
    log_max_bytes: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env (like VITE_API_URL for frontend)
        
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> any:
            if field_name in ["qdrant_port", "api_port", "vector_size", "face_min_size", 
                             "log_max_bytes", "log_backup_count"]:
                return int(raw_val)
            return raw_val
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self._ensure_directories()
        self._set_model_env_vars()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        dirs = [
            self.models_dir,
            self.log_dir,
            self.raw_dir,
            self.processing_dir,
            self.processed_dir,
            self.thumbnail_dir,
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _set_model_env_vars(self):
        """Set environment variables for model storage."""
        models_path = Path(self.models_dir).resolve()
        insightface_dir = models_path / "insightface"
        sentence_transformers_dir = models_path / "sentence_transformers"
        huggingface_dir = models_path / "huggingface"
        
        os.environ["INSIGHTFACE_ROOT"] = str(insightface_dir)
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(sentence_transformers_dir)
        os.environ["HF_HOME"] = str(huggingface_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(huggingface_dir)
    
    @property
    def insightface_dir(self) -> Path:
        return Path(self.models_dir) / "insightface"
    
    @property
    def sentence_transformers_dir(self) -> Path:
        return Path(self.models_dir) / "sentence_transformers"
    
    @property
    def huggingface_dir(self) -> Path:
        return Path(self.models_dir) / "huggingface"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

