# Vision Cap Backend Architecture

## Overview

The backend has been refactored into a modular, extensible architecture following best practices. This allows for easy experimentation with different models and features.

## Directory Structure

```
backend/
├── config/              # Configuration management
│   ├── __init__.py
│   └── settings.py      # Application settings (Pydantic)
├── core/                # Core components
│   ├── __init__.py
│   ├── logging_config.py
│   └── models/          # Model abstractions
│       ├── __init__.py
│       ├── base.py      # Abstract base classes
│       ├── face_detection.py
│       ├── embeddings.py
│       ├── vlm.py
│       └── factory.py   # Model factory
├── services/            # Business logic layer
│   ├── __init__.py
│   ├── storage_service.py    # Qdrant operations
│   ├── image_processor.py     # Image processing pipeline
│   └── search_service.py      # Search operations
├── api/                 # API layer
│   ├── __init__.py
│   ├── schemas.py       # Pydantic models
│   └── routes/          # API routes
│       ├── __init__.py
│       ├── search.py
│       ├── feed.py
│       └── stats.py
├── utils/               # Utility functions
│   ├── __init__.py
│   └── image_utils.py
├── main.py              # FastAPI application
└── worker.py            # Background worker

```

## Key Design Patterns

### 1. **Abstract Base Classes**
All models implement abstract interfaces:
- `BaseFaceDetectionModel` - Face detection interface
- `BaseEmbeddingModel` - Embedding generation interface
- `BaseVLM` - Vision Language Model interface

### 2. **Factory Pattern**
`ModelFactory` creates model instances based on configuration:
```python
face_model = ModelFactory.create_face_detection_model()
embedding_model = ModelFactory.create_embedding_model()
vlm = ModelFactory.create_vlm()
```

### 3. **Dependency Injection**
Services receive dependencies through constructors:
```python
search_service = SearchService(
    face_model=face_model,
    embedding_model=embedding_model,
    storage_service=storage_service
)
```

### 4. **Service Layer**
Business logic is separated into services:
- `StorageService` - Database operations
- `ImageProcessor` - Image processing pipeline
- `SearchService` - Search operations

### 5. **Configuration Management**
Centralized configuration using Pydantic Settings:
```python
from config import get_settings
settings = get_settings()
```

## Adding New Models

### Adding a New Face Detection Model

1. Create a new class in `core/models/face_detection.py`:
```python
class NewFaceModel(BaseFaceDetectionModel):
    def load(self) -> None:
        # Load model
        pass
    
    def detect_faces(self, image: np.ndarray) -> List[np.ndarray]:
        # Detect and return embeddings
        pass
    
    def get_face_count(self, image: np.ndarray) -> int:
        # Return face count
        pass
```

2. Update `ModelFactory.create_face_detection_model()`:
```python
if model_type == "new_model":
    return NewFaceModel()
```

3. Update `config/settings.py`:
```python
face_detection_model: str = "new_model"
```

### Adding a New Embedding Model

1. Create a new class in `core/models/embeddings.py`:
```python
class NewEmbeddingModel(BaseEmbeddingModel):
    def load(self) -> None:
        pass
    
    def encode_image(self, image) -> np.ndarray:
        pass
    
    def encode_text(self, text: str) -> np.ndarray:
        pass
    
    @property
    def embedding_size(self) -> int:
        return 512
```

2. Update factory and config similarly.

### Adding a New VLM

1. Create a new class in `core/models/vlm.py`:
```python
class NewVLM(BaseVLM):
    def load(self) -> None:
        pass
    
    def generate_tags(self, image, **kwargs) -> List[str]:
        pass
    
    def generate_caption(self, image, **kwargs) -> Optional[str]:
        pass
```

2. Update factory and config.

## Configuration

All configuration is managed through environment variables and `config/settings.py`:

- `QDRANT_HOST` - Qdrant host
- `QDRANT_PORT` - Qdrant port
- `MODELS_DIR` - Models directory
- `LOG_DIR` - Logs directory
- `FACE_DETECTION_MODEL` - Face detection model type
- `EMBEDDING_MODEL` - Embedding model type
- `VLM_MODEL` - VLM model type

## Benefits

1. **Modularity** - Clear separation of concerns
2. **Extensibility** - Easy to add new models
3. **Testability** - Services can be easily mocked
4. **Maintainability** - Well-organized code structure
5. **Type Safety** - Pydantic models and type hints
6. **Configuration** - Centralized settings management

## Migration Notes

- Old `models.py` is preserved as reference
- Old `main.py` and `worker.py` are backed up as `*_old.py`
- API endpoints remain the same - no breaking changes
- Behavior is consistent with previous version

