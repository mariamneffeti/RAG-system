---
title: Rag System
emoji: 🔍
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 5000
pinned: false
---

# RAG System

A full-stack Retrieval-Augmented Generation system. Upload documents, ask questions, get grounded answers powered by semantic search and Groq's free LLM API — streamed in real time.

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Flask](https://img.shields.io/badge/Flask-3.0-green) ![Groq](https://img.shields.io/badge/LLM-Groq-orange) ![FAISS](https://img.shields.io/badge/VectorDB-FAISS-red) [![HuggingFace Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/mariammmnefff/rag-system)

**🚀 [Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/mariammmnefff/rag-system)**

---

## How it works

```
Documents → Chunking → Embeddings → FAISS Index → Retrieval → Groq LLM → Streamed Answer
```

1. **Load** — Upload `.txt`, `.md`, `.pdf`, or `.docx` files
2. **Chunk** — Split into overlapping sentence-boundary segments
3. **Embed** — `paraphrase-MiniLM-L3-v2` converts each chunk to a 384-dim vector
4. **Index** — FAISS stores vectors for millisecond cosine similarity search
5. **Retrieve** — Top-K most relevant chunks fetched for any query
6. **Generate** — Groq streams a grounded answer token by token

---

## Project structure

```
RAG/
├── static/
│   ├── index.html      # Landing page
│   ├── ingest.html     # Document upload UI
│   └── query.html      # Chat/query UI
├── engine.py           # Core RAG pipeline
├── server.py           # Flask REST API + SSE streaming
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env                # Your secrets (never committed)
├── .env.example        # Template for other developers
└── .gitignore
```

---

## Quick start (local)

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/rag-system.git
cd rag-system
pip install -r requirements.txt
```

### 2. Get a free Groq API key

Sign up at [console.groq.com](https://console.groq.com) → API Keys → Create Key

### 3. Set your key

```bash
# Create .env file
cp .env.example .env
# Edit .env and add your key
```

**Windows PowerShell:**
```powershell
$env:GROQ_API_KEY = "gsk_..."
python server.py
```

**Mac/Linux:**
```bash
export GROQ_API_KEY="gsk_..."
python server.py
```

### 4. Open the app

```
http://localhost:5000
```

---

## Deploy with Docker

### Build and run

```bash
docker build -t rag-system .
docker run -p 5000:5000 -e GROQ_API_KEY=gsk_... rag-system
```

### Or use Docker Compose

```bash
# Add your key to .env first, then:
docker-compose up
```

Open `http://localhost:5000`

---

## Deploy to Hugging Face Spaces

This app is deployed on **Hugging Face Spaces** using Docker. HF Spaces provides 16GB of free RAM which is necessary since `faiss-cpu` + `sentence-transformers` exceed the 512MB limit of most free hosting platforms.

### Steps to deploy your own copy

1. Go to **huggingface.co** → New Space → SDK: **Docker**
2. Push your code:
```bash
git remote add hf https://YOUR_USERNAME:YOUR_HF_TOKEN@huggingface.co/spaces/YOUR_USERNAME/rag-system
git push hf main --force
```
3. Add your secret: Space **Settings** → **Variables and secrets** → `GROQ_API_KEY = gsk_...`
4. HF builds and deploys automatically from your `Dockerfile`

> The space sleeps after 48 hours of inactivity but wakes up quickly on the next visit.

---

## Changing the embedding model

The default model is `paraphrase-MiniLM-L3-v2` — lightweight (22MB), chosen specifically because free deployment platforms cap memory at 512MB and heavier models exceed that limit.

To swap it, open `server.py` and change `model_name`:

```python
rag = RAGPipeline(
    model_name="all-MiniLM-L6-v2",  # change this
    chunk_size=512,
    chunk_overlap=64,
)
```

Available models ranked by size vs quality:

| Model | Size | RAM needed | Best for |
|-------|------|-----------|---------|
| `paraphrase-MiniLM-L3-v2` | 22MB | ~512MB | Deployment (default) |
| `all-MiniLM-L6-v2` | 90MB | ~1GB | Local use — better quality |
| `all-MiniLM-L12-v2` | 120MB | ~1.5GB | Local use — even better |
| `all-mpnet-base-v2` | 420MB | ~2GB | Local use — best quality |

> ⚠️ If you switch models after ingesting documents, delete the `rag_store/` folder and re-ingest everything. Vectors from different models are incompatible.

---

## Running locally with custom parameters

Clone the repo, install dependencies, then configure `server.py` to your liking:

```python
rag = RAGPipeline(
    model_name="all-MiniLM-L6-v2",   # embedding model (see table above)
    chunk_size=512,                    # max characters per chunk (try 256–1024)
    chunk_overlap=64,                  # overlap between chunks (try 32–128)
)
```

**What each parameter does:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `model_name` | `paraphrase-MiniLM-L3-v2` | Which embedding model to use |
| `chunk_size` | `512` | Larger = more context per chunk, less precise retrieval |
| `chunk_overlap` | `64` | Higher = smoother chunk boundaries, more storage |

**Set your API key (choose one method):**

Option 1 — `.env` file (recommended):
```
GROQ_API_KEY=gsk_...
```

Option 2 — PowerShell (session only):
```powershell
$env:GROQ_API_KEY = "gsk_..."
python server.py
```

Option 3 — paste directly in the UI Settings panel at `http://localhost:5000/query`

---

## Deployment note

This app uses `faiss-cpu` + `sentence-transformers` which together require **more than 512MB of RAM** to load. Render's free tier has a 512MB limit, which caused out-of-memory errors during deployment even with the lightest model.

The app is therefore deployed on **Hugging Face Spaces** which provides **16GB of free RAM** and is built specifically for ML applications like this one.

🚀 **[Live demo](https://huggingface.co/spaces/mariammmnefff/rag-system)**

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Landing page |
| `GET` | `/ingest` | Ingest UI |
| `GET` | `/query` | Query UI |
| `POST` | `/ingest/text` | Ingest raw text `{ text, source }` |
| `POST` | `/ingest/file` | Upload file (multipart) |
| `GET` | `/query/stream?q=...&k=5` | SSE streaming query |
| `GET` | `/retrieve?q=...` | Raw chunk results (JSON) |
| `GET` | `/stats` | Index statistics |
| `POST` | `/clear` | Wipe the index |

### Example: ingest text

```bash
curl -X POST http://localhost:5000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document content here", "source": "my-doc.txt"}'
```

### Example: query

```bash
curl "http://localhost:5000/retrieve?q=what+is+RAG&k=3"
```

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Free key from console.groq.com |

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Embeddings | `sentence-transformers` — all-MiniLM-L6-v2 |
| Vector DB | `faiss-cpu` — IndexFlatIP (cosine similarity) |
| LLM | Groq API — llama-3.3-70b-versatile (free tier) |
| Backend | Flask + Flask-CORS |
| Streaming | Server-Sent Events (SSE) |
| Frontend | Vanilla HTML/CSS/JS |

---

## License

MIT