"""
RAG Flask Server - Rest API + SSE streaming for the web interface.

Run:
export ANTHROPIC_API_KEY=sk-...
python server.py

Endpoints :
POST /ingest/text        { "text": "...", "source": "..." }
    POST /ingest/file        multipart/form-data  file=<file>
    GET  /query/stream?q=... SSE stream
    GET  /stats
    POST /clear
"""

#imports
import os
import json
import tempfile
from pathlib import Path
from flask import Flask,request,jsonify,Response,stream_with_context,send_from_directory
from flask_cors import CORS
from engine import RAGPipeline,load_file
from dotenv import load_dotenv
load_dotenv()

STATIC_DIR = Path(__file__).parent / "static"
app = Flask(__name__,static_folder=str(STATIC_DIR),static_url_path="/static")
CORS(app)

#single global pipeline instance(swap for per_user sessions in prod)
rag = RAGPipeline(
    model_name="paraphrase-MiniLM-L3-v2",
    chunk_size=512,
    chunk_overlap=64,
)

ALLOWED_EXTENSIONS = {".txt",".md",".pdf",".docx"}
STORE_PATH = "./rag_store"
#page routes
@app.route("/")
def home():
    return send_from_directory(STATIC_DIR, "index.html")
 
@app.route("/ingest")
def ingest_page():
    return send_from_directory(STATIC_DIR, "ingest.html")
 
@app.route("/query")
def query_page():
    return send_from_directory(STATIC_DIR, "query.html")
#ingest

@app.route("/ingest/text",methods=["POST"])
def ingest_text():
    data = request.get_json(force=True)
    text = data.get("text","").strip()
    source = data.get("source","inline")
    if not text :
        return jsonify({"error":"No text provided"}),400
    rag.ingest_text(text,source=source)
    rag.save(STORE_PATH)
    return jsonify({"status":"ok","stats": rag.stats})
@app.route("/ingest/file",methods=["POST"])
def ingest_file():
    if "file" not in request.files :
        return jsonify({"error":"No file in request"}),400
    f = request.files["file"]
    suffix = Path(f.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type : {suffix}"}),400
    with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name
    try:
        rag.ingest_file(tmp_path)
        rag.save(STORE_PATH)
    finally:
        os.unlink(tmp_path)
    return jsonify({"status" : "ok", "filename" : f.filename,"stats" : rag.stats})
# Query
@app.route("/query/stream")
def query_stream():
    question = request.args.get("q","").strip()
    top_k = int(request.args.get("k",5))
    api_key = request.headers.get("X-API-KEY","") or os.environ.get("GROQ_API_KEY", "")
    if not question:
        return jsonify({"error":"No query provided"}),400
    def generate():
        try:
            for chunk in rag.query_stream(question,top_k=top_k,api_key=api_key):
                yield chunk
        except Exception as e:
            yield f"data: {json.dumps({'error':str(e)})}\n\n"
            yield "data: [DONE]\n\n"
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control" : "no-cache",
            "X-Accel-Buffering": "no",
        },
        )

@app.route("/retrieve")
def retrieve():
    question = request.args.get("q","").strip()
    top_k = int(request.args.get("k",5))
    if not question:
        return jsonify({"error":"no query"}),400
    results = rag.retrieve(question,top_k=top_k)
    return jsonify([
        {
            "score" : round(r.score,4),
            "text":r.chunk.text,
            "source": r.chunk.metadata.get("source","?"),
            "chunk_index" : r.chunk.chunk_index,
        }
        for r in results
    ])

#management

@app.route("/stats")
def stats():
    return jsonify(rag.stats)

@app.route("/chunks")
def list_chunks():
    page = int(request.args.get("page",0))
    size = int(request.args.get("size",20))
    chunks = rag.vector_store.chunks
    total = len(chunks)
    page_chunks = chunks[page * size : (page+1)*size]
    return jsonify({
        "total" : total,
        "page" : page,
        "chunks" : [
            {"id" : c.chunk_id,"text" : c.text[:200],"source" : c.metadata.get("source","?")}
        for c in page_chunks
        ],
    })

@app.route("/clear",methods=["POST"])
def clear():
    global rag
    rag = RAGPipeline()
    import shutil
    if Path(STORE_PATH).exists():
        shutil.rmtree(STORE_PATH)
    return jsonify({"status" : "cleared"})

if __name__ == "__main__" :
    #load persisted store if it exists
    if Path(f"{STORE_PATH}/index.faiss").exists() :
        print("[SERVER] LOADING PERSISTED VECTOR STORE...")
        rag.load(STORE_PATH)
    port = int(os.environ.get("PORT", 7860)) 
    
    print(f"[SERVER] Starting on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)