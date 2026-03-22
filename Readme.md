# RAG System

A full-stack Retrieval-Augmented Generation system. Upload documents, ask questions, get grounded answers powered by semantic search and Groq's free LLM API — streamed in real time.

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Flask](https://img.shields.io/badge/Flask-3.0-green) ![Groq](https://img.shields.io/badge/LLM-Groq-orange) ![FAISS](https://img.shields.io/badge/VectorDB-FAISS-red)

---

## How it works

```
Documents → Chunking → Embeddings → FAISS Index → Retrieval → Groq LLM → Streamed Answer
```

1. **Load** — Upload `.txt`, `.md`, `.pdf`, or `.docx` files
2. **Chunk** — Split into overlapping sentence-boundary segments
3. **Embed** — `all-MiniLM-L6-v2` converts each chunk to a 384-dim vector
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
├── vercel.json
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

## Deploy to Vercel

> **Note:** Vercel is designed for frontend/serverless apps. For a full Python Flask backend, use **Railway**, **Render**, or **Fly.io** instead (all have free tiers). The `vercel.json` included here uses Vercel's Python serverless runtime.

```bash
npm install -g vercel
vercel login
vercel --prod
```

When prompted, add your environment variable:
```
GROQ_API_KEY = gsk_...
```

Or set it in the Vercel dashboard: **Project → Settings → Environment Variables**

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