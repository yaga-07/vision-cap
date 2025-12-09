FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV and image processing
# Note: libgl1-mesa-glx is deprecated in Debian Trixie, using libgl1 instead
# Also install build tools needed for compiling insightface Cython extensions
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    g++ \
    make \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Note: images/, models/, and qdrant_storage/ are mounted as volumes
# Do not create them here to avoid duplication

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

