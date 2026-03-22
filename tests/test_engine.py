"""
Basic tests for the RAG engine.
Run: pytest tests/ -v
"""

from engine import (
    Document,
    Chunk,
    load_text,
    chunk_document,
    RAGPipeline,
)


def test_document_creation():
    doc = load_text("Hello world.", source_name="test.txt")
    assert doc.content == "Hello world."
    assert doc.metadata["source"] == "test.txt"
    assert doc.doc_id != ""


def test_document_same_content_same_id():
    doc1 = load_text("Same content")
    doc2 = load_text("Same content")
    assert doc1.doc_id == doc2.doc_id


def test_chunking_produces_chunks():
    doc = load_text("This is sentence one. This is sentence two. This is sentence three.")
    chunks = chunk_document(doc, chunk_size=50)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_metadata():
    doc = load_text("Some text here.", source_name="myfile.txt")
    chunks = chunk_document(doc)
    assert chunks[0].doc_id == doc.doc_id
    assert chunks[0].metadata["source"] == "myfile.txt"


def test_pipeline_ingest_and_retrieve():
    rag = RAGPipeline()
    rag.ingest_text(
        "FAISS is a library for efficient similarity search.",
        source="test"
    )
    results = rag.retrieve("What is FAISS?", top_k=1)
    assert len(results) == 1
    assert results[0].score > 0


def test_pipeline_stats():
    rag = RAGPipeline()
    rag.ingest_text("Some document content.", source="test")
    stats = rag.stats
    assert stats["documents"] == 1
    assert stats["chunks"] >= 1


def test_empty_query_returns_nothing():
    rag = RAGPipeline()
    results = rag.retrieve("anything", top_k=5)
    assert results == []