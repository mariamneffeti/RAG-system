"""
RAG Engine - Document loader, chunking,embeddings, FAISS vector DB, query pipeline.
"""

#imports
import os
import re
import json
import pickle
import hashlib
from pathlib import Path
from dataclasses import dataclass,field
from typing import Iterator
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

#Data Structures : document,chunk,searchresult


@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""
    def __post_init__(self):
        if not self.doc_id :
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12] #if there is no id created a unique one depending on the content(same content same id,different content different id)

@dataclass
class Chunk:
    text:str
    chunk_id:str
    doc_id:str
    chunk_index: int
    metadata : dict = field(default_factory=dict)

@dataclass
class SearchResult:
    chunk: Chunk
    score : float

#helper functions: document loaders

def load_txt(path : str) -> Document :
    with open(path,r,encoding="utf-8",errors="replace") as f:
        content = f.read()
    return Document(content=content,metadata={"source" :path,"type" :"txt"}) #id generated randomly

def load_pdf(path:str) -> Document:
    try:
        import PyPDF2
        text=[]
        with open(path,"rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
            content = "\n".join(text)
    except ImportError:
        raise ImportError("PyPDF2 required for PDF loading: pip install PyPDF2")
    return Document(content=content,metadata={"source":path,"type":"pdf"})

def load_docx(path:str) -> Document :
    try :
        from docx import Document as DocxDocument
        doc = DocxDocument(path)
        content = "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        raise ImportError("python-docx required: pip install python-docx")
    return Document(content=content,metadata={"source":path,"type":"docx"})

def load_file(path:str) -> Document :
    ext = Path(path).suffix.lower()
    loaders = {".txt": load_txt,".pdf": load_pdf,".docx":load_docx,".md":load_txt}
    loader = loaders.get(ext)
    if not loader:
        raise ValueError(f"Unsupported file type : {ext}")
    return loader(path)
def load_text(text : str, source_name : str = "inline") -> Document:
    return Document(content=text,metadata={"source":source_name,"type":"text"})

def load_Directory(dir_path : str) -> list[Document]:
    docs = []
    for p in Path(dir_path).rglob("*"):
        if p.suffix.lower() in (".txt",".md",".pdf",".docx"):
            try:
                docs.append(load_file(str(p)))
            except Exception as e:
                print(f"!! skipping {p}: {e}")
    return docs

#helper functions : chunking

def _merge_sentences(sentences : list[str],max_chars : int) -> list[str]:
    segments,current = [],[]
    current_len =0
    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current :
            segments.append(" ".join(current))
            current, current_len = [] , 0
        current.append(sentence)
        current_len += len(sentence)
    if current:
        segments.append(" ".join(current))
    return segments

def _fixed_chunks(text : str, size : int, overlap : int) -> list[str]:
    chunks,start = [],0
    while start < len(text):
        end = min(start + size,len(text))
        chunks.append([text[start:end]])
        start += size - overlap
    return chunks

def chunk_document(
    doc : Document,
    chunk_size : int = 512,
    chunk_overlap : int =64,
    strategy : str = "sentence",
)-> list[Chunk]:
    """
    Split a document into overlapping chunks
    the strategy used is sentence as to not lose meaning -> better sematic meaning
    the user can chose 'fixed' strategy if he want faster results.
    """
    text = doc.content.strip()
    if not text:
        return []
    if strategy == "sentence" :
        sentences = re.split(r"(?<=[.!?])\s+",text) #logic to know a sentence ; where it ends
        segments = _merge_sentences(sentences,chunk_size)
    else:
        segments = _fixed_chunks(text,chunk_size,chunk_overlap)
    chunks = []
    for i,seg in enumerate(segments):
        chunk_id = f"{doc.doc_id}-{i}"
        chunks.append(Chunk(
            text=seg,
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            chunk_index=i,
            metadata={**doc.metadata,"chunk_index":i,"total_chunks":len(segments)}
        )
        )
    return chunks

# embeddings

class EmbeddingModel:
    def __init__(self,model_name : str = "all-MiniLM-L6-v2"):
        print(f"~~Loding Model Name :{model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = self.model.get_sentence_embedding_dimension() #in case of switching models
        print(f" Embedding Dimension : {self.dim}")
        
    def embed(self, texts: list[str],batch_size : int = 64) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar = len(texts) > 100,
            normalize_embeddings = True,
            )
        return vecs.astype(np.float32)
        
    def embed_one(self,text:str) -> np.ndarray :
        return self.embed([text])[0]

# FAISS Vector Store

class VectorStore:
    def __init__(self,dim : int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []
    def add(self, chunks : list[Chunk], embeddings: np.ndarray):
        assert len(chunks) == len(embeddings) , "chunk/embedding count mismatch"
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        print(f"Added {len(chunks)} chunks. Total: {len(self.chunks)}")

    def search(self,query_vec:np.ndarray,top_k:int =5) -> list[SearchResult]:
        if self.index.ntotal == 0:
            return []
        q = query_vec.reshape(1,-1)
        scores, indices = self.index.search(q,min(top_k,self.index.ntotal))
        results = []
        for score,idx in zip(scores[0],indices[0]):
            if idx >= 0 :
                results.append(SearchResult(chunk=self.chunks[idx],score=float(score)))
        return results

    def save(self,path : str):
        Path(path).mkdir(parents=True,exist_ok=True)
        faiss.write_index(self.index,f"{path}/index.faiss")
        with open(f"{path}/chunks.pkl",wb) as f :
            pickle.dump(self.chunks,f)
        with open(f"{path}/meta.json","w") as f :
            json.dump({"dim":self.dim,"total":len(self.chunks)},f)
        print("Saved to {path}")


    @classmethod
    def load(cls,path: str) -> "VectorStore":
        with open(f"{path}/meta.json") as f :
            meta = json.load(f)
        vs = cls(dim = meta["dim"])
        with open(f"{path}/chunks.pkl","rb") as f:
            vs.chunks = pickle.load(f)
        print(f"Loaded {len(vs.chunks)} chunks from {path}")
        return vs
# RAG pipeline
class RAGPipeline:
    def __init__(
        self,
        model_name:str = "all-MiniLM-L6-v2",
        chunk_size:int =512,
        chunk_overlap:int = 64,
        chunk_stategy : str = "sentence"):

        self.embedder = EmbeddingModel(model_name)
        self.vector_store = VectorStore(self.embedder.dim)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_stategy = chunk_stategy
        self.docs: list[Document]=[]
    #Ingestion

    def ingest(self,docs:list[Document]):
        all_chunks = []
        for doc in docs:
            self.docs.append(doc)
            chunks = chunk_document(
                doc,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                strategy=self.chunk_stategy,
            )
            all_chunks.extend(chunks)
        if not all_chunks :
            print("[WARNING] NO CHUNKS PRODUCED")
            return
        texts = [c.text for c in all_chunks]
        print(f"[RAG] Embedding {len(texts)} chunks...")
        embeddings = self.embedder.embed(texts)
        self.vector_store.add(all_chunks,embeddings)
    
    def ingest_file(self,path: str):
        self.ingest(load_file(path=path))

    def ingest_text(self,text:str,source:str = "inline"):
        self.ingest([load_text(text,source)])
    
    def ingest_Directory(self,path : str):
        self.ingest(load_Directory(path))
    
    #Retrieval
    def retrieve(self,query : str, top_k: int =5) -> list[SearchResult]:
        q_vec = self.embedder.embed_one(query)
        return self.vector_store.search(q_vec,top_k=top_k)
    #building context
    def build_context(self,results:list[SearchResult],max_chars:int =3000) -> str:
        parts,total = [],0
        for r in results :
            snippet = r.chunk.text.strip()
            source = r.chunk.metadata.get("source","unkown")
            block = f"[Source :{source} | Score : {r.score : .3f}]\n{snippet}"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n\n---\n\n".join(parts)
    #Streaming Query
    def query_stream(self,question : str, top_k: int =5, api_key: str = None) -> Iterator[str] :
        """
        Retrieve context -> stream an answer from Anthropic API.
        Yields text chunks as they arrive : SSE-friendly"""
        results = self.retrieve(question,top_k=top_k)
        context = self.build_context(results)
        if not context:
            yield " NO relevant documents found, Please ingest some documents first."
            return
        yield f"Data : __Sources__{json.dumps([{'source' : r.chunk.metadata.get('source','?'),'score': round(r.score,3), 'preview': r.chunk.text[:120]} for r in results])}\n\n"
        import anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY","")
        client = anthropic.Anthropic(api_key=key)
        system = (
            "You are a helpful assistant. Answer questions using ONLY the provided context."
            "If the answer isn't in the context,say so. Be concise and accurate"
        )
        user_msg = f"Context:\n{context}\n\nQestion:{question}"
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system,
            messages=[{"role":"user","content":user_msg}],
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {json.dumps(text)}\n\n"
        yield "data: [DONE]\n\n"

    #Persistence

    def save(self,path : str = "./rag_store"):
        self.vector_store.save(path)
    def load(self,path:str ="./rag_store"):
        self.vector_store = VectorStore.load(path)
    @property
    def stats(self) -> dict :
        return {
            "documents" : len(self.docs),
            "chunks" : len(self.vector_store.chunks),
            "embedding_model" : self.embedder.model_name,
            "embedding_dim" : self.embedder.dim,
        }


#CLI Demo
     
if __name__ == "__main__":
    rag = RAGPipeline()
 
    sample = """
    Retrieval-Augmented Generation (RAG) is a technique that enhances large language models
    by giving them access to external knowledge bases at inference time. Instead of relying
    solely on parameters learned during training, RAG retrieves relevant documents and feeds
    them as context into the LLM prompt.
 
    The core RAG pipeline consists of:
    1. Document ingestion: Loading and parsing source documents.
    2. Chunking: Splitting documents into smaller overlapping segments.
    3. Embedding: Converting text chunks into dense vector representations.
    4. Vector store: Indexing embeddings for fast similarity search (e.g., FAISS).
    5. Retrieval: Finding the top-k most relevant chunks for a given query.
    6. Generation: Using retrieved context + LLM to produce a grounded answer.
 
    FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and
    clustering of dense vectors. It supports billion-scale vector indices and various index
    types including flat, IVF, HNSW, and PQ.
 
    Sentence Transformers are pre-trained models that produce semantically meaningful
    sentence embeddings. The all-MiniLM-L6-v2 model produces 384-dimensional embeddings
    and balances speed with quality, making it ideal for RAG systems.
    """
 
    print("=== Ingesting sample document ===")
    rag.ingest_text(sample, source="rag_overview.txt")
    print(f"Stats: {rag.stats}\n")
 
    queries = [
        "What is RAG?",
        "What is FAISS used for?",
        "How does chunking work?",
    ]
 
    for q in queries:
        print(f"\nQ: {q}")
        results = rag.retrieve(q, top_k=2)
        for r in results:
            print(f"  [{r.score:.3f}] {r.chunk.text[:100]}...")
 