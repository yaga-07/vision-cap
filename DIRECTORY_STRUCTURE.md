# Directory Structure

This document explains the correct directory structure and why certain directories should not be duplicated.

## Correct Structure

```
vision_cap/
├── backend/              # Backend Python code only
│   ├── main.py
│   ├── models.py
│   ├── worker.py
│   └── ...              # NO images/, models/, qdrant_storage/ here!
│
├── frontend/            # Frontend React code only
│   └── ...
│
├── images/              # ✅ Image storage (root level)
│   ├── raw/            # Drop photos here for processing
│   ├── processing/     # Temporary processing queue
│   ├── processed/      # Processed full-size images
│   └── thumbnails/     # Generated thumbnails
│
├── models/              # ✅ Model weights (root level)
│   ├── insightface/
│   ├── sentence_transformers/
│   └── huggingface/
│
└── qdrant_storage/      # ✅ Database files (root level)
    └── collections/
```

## Why This Structure?

### Volume Mounts in Docker

The `docker-compose.yml` mounts root-level directories into containers:

```yaml
volumes:
  - ./backend:/app                    # Backend code
  - ./images:/app/images             # Images from root
  - ./models:/app/models             # Models from root
  - ./qdrant_storage:/app/qdrant_storage  # DB from root
```

### The Problem with Duplicates

If `backend/images/`, `backend/models/`, or `backend/qdrant_storage/` exist:
1. They create confusion about which directory is used
2. They can cause data loss if code writes to the wrong location
3. They waste disk space
4. They can cause permission issues

### Container View

Inside containers, the structure looks like:
```
/app/                    # Mounted from ./backend
├── main.py
├── models.py
├── images/              # Mounted from ./images (root)
├── models/              # Mounted from ./models (root)
└── qdrant_storage/     # Mounted from ./qdrant_storage (root)
```

## Cleaning Up Duplicates

If you have duplicate directories in `backend/`, run:

```bash
./scripts/cleanup_duplicates.sh
```

Or manually remove them:
```bash
rm -rf backend/images/
rm -rf backend/models/
rm -rd backend/qdrant_storage/
```

## Prevention

The `backend/.gitignore` file prevents these directories from being committed:
- `images/`
- `models/`
- `qdrant_storage/`

## Data Locations

| Data Type | Root Directory | Container Path | Purpose |
|-----------|---------------|----------------|---------|
| Images | `./images/` | `/app/images/` | Photo storage |
| Models | `./models/` | `/app/models/` | AI model weights |
| Database | `./qdrant_storage/` | `/app/qdrant_storage/` | Qdrant DB files |

All data should be in root-level directories, never in `backend/` or `frontend/`.

