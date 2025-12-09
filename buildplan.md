Here is the build plan for your **Local Event Photo Search System**.

This plan prioritizes **speed of development** (using pre-made libraries), **speed of execution** (running locally on a laptop), and **privacy** (100% offline).

-----

### **Phase 1: The Infrastructure (The "Box")**

*Goal: Create a portable, isolated environment that runs anywhere without internet.*

**1. Docker Architecture**
We will use Docker Compose to spin up three containers. This ensures the photographer can just install Docker and run one command to start everything.

  * **Container A: Vector Database (Qdrant)**
      * *Why:* Lightweight, Rust-based, blazingly fast. Stores face embeddings and text embeddings.
      * *Image:* `qdrant/qdrant`
  * **Container B: Backend (Python/FastAPI)**
      * *Why:* Handles image processing, AI models, and API requests.
      * *Hardware Access:* Needs GPU access (pass-through) if available, or optimized CPU libraries (ONNX).
  * **Container C: Frontend (React/Nginx)**
      * *Why:* Serves the Web App to guests. Nginx handles static files (the actual photos).

**2. Network Setup (The "Hassle-Free" Link)**

  * **mDNS (Bonjour):** Configure the server to broadcast `http://photos.local`.
  * **Hotspot:** The laptop creates a Wi-Fi hotspot named "EventPhotos". Guests connect to this, or the venue router.

-----

### **Phase 2: The AI Processing Worker (The Core)**

*Goal: A background process that watches a folder and processes images in \<3 seconds.*

**1. The "Watchdog" Script**

  * Use the Python `watchdog` library to monitor a folder `  /images/raw `.
  * When a file triggers `on_created`, move it to a `processing` queue.

**2. The Logic Pipeline (Step-by-Step)**

  * **Step A: Face Indexing (InsightFace)**

      * *Library:* `insightface` (Model: `buffalo_l`).
      * *Action:* Detect faces. For each face, extract the **512-dim embedding**.
      * *Optimization:* Only process faces larger than 40x40 pixels to avoid background noise.
      * *Grouping Logic:*
          * If `face_count == 1` $\rightarrow$ Metadata: `type: solo`
          * If `face_count == 2` $\rightarrow$ Metadata: `type: duo`
          * If `face_count > 2` $\rightarrow$ Metadata: `type: group`

  * **Step B: Semantic Indexing (CLIP)**

      * *Library:* `sentence-transformers` (Model: `clip-ViT-B-32`).
      * *Action:* Embed the *whole image* into a vector.
      * *Result:* Enables searches like "dancing" or "red dress" without text tags.

  * **Step C: Metadata Extraction (Florence-2)**

      * *Library:* `transformers` (Model: `Florence-2-large`).
      * *Task 1:* Run `<OD>` (Object Detection). This returns a list of objects/boxes.
          * *Extract:* List of labels (e.g., `["cake", "wine glass", "suit"]`). Save this as JSON tags.
      * *Task 2:* Run `<DETAILED_CAPTION>`.
          * *Extract:* "A bride and groom cutting a white cake." Save this as a text field for keyword search.

  * **Step D: Database Insert (Qdrant)**

      * **Point 1 (The Image):** ID=PhotoID, Vector=CLIP\_Vector, Payload={JSON Metadata, Caption}.
      * **Point 2..N (The Faces):** ID=UUID, Vector=Face\_Vector, Payload={PhotoID, ParentImage}.

-----

### **Phase 3: The API (FastAPI)**

*Goal: Connect the frontend to the database.*

**Endpoints Needed:**

1.  `POST /search/face`: Accepts a selfie (blob).
      * Runs InsightFace on the selfie $\rightarrow$ Gets Vector.
      * Queries Qdrant for matching Face Vectors (Cosine Similarity \> 0.6).
      * Returns list of `ParentImage` URLs.
2.  `POST /search/text`: Accepts text string.
      * Runs CLIP on text $\rightarrow$ Gets Vector.
      * Queries Qdrant for Image Vectors.
3.  `GET /feed`: Returns all images (paginated) for the "Live Feed".

-----

### **Phase 4: The User Interface (React + PWA)**

*Goal: A mobile-first, app-like experience.*

**1. Guest View**

  * **Landing:** Large button "Find Me" (activates camera) + Search Bar.
  * **Results Grid:** Masonry layout (Pinterest style).
  * **Download:** Long-press to download high-res.

**2. Photographer View (Dashboard)**

  * **Project Manager:** Create "New Event" (Generates a new Qdrant Collection).
  * **Live Stats:** "Images Processed: 500", "Unique Faces: 120".
  * **QR Code Generator:** Displays the QR code for `http://photos.local`.

-----

### **Implementation Steps (Cheat Sheet)**

**Step 1: Set up the Database**
Create a `docker-compose.yml` file to start Qdrant.

```yaml
version: '3'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_storage:/qdrant/storage
```

**Step 2: Create the Ingestion Script**
Create `worker.py`. Load models *outside* the loop (Global variables) so they stay in RAM.

```python
# Pseudo-code for worker.py
from insightface.app import FaceAnalysis
from sentence_transformers import SentenceTransformer

# Load models once
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0) # 0 for GPU, -1 for CPU
clip_model = SentenceTransformer('clip-ViT-B-32')

def process_image(path):
    img = cv2.imread(path)
    # 1. Get Faces
    faces = face_app.get(img)
    for face in faces:
        store_face_in_qdrant(face.embedding, path)
    
    # 2. Get Semantic Vector
    clip_vec = clip_model.encode(Image.open(path))
    store_image_in_qdrant(clip_vec, path)
```

**Step 3: Build the API**
Create `main.py` with FastAPI to serve the data stored by `worker.py`.

**Step 4: The Frontend**
Use `Vite` to scaffold a React app. Use `react-webcam` for the selfie capture.