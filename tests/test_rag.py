"""Tests for RAG pipeline components."""
import pytest
import numpy as np
from app.rag.embeddings import EmbeddingModel, compute_similarity
from app.rag.chunk_documents import chunk_text, chunk_by_paragraphs, TextChunk
from app.rag.vector_store import VectorStore
from app.rag.retriever import RAGRetriever


def test_chunk_text():
    """Test text chunking with fixed size."""
    text = "This is a test. " * 100
    chunks = chunk_text(text, source="test_doc", chunk_size=100, overlap=20)
    
    assert len(chunks) > 0
    assert all(chunk.source == "test_doc" for chunk in chunks)
    assert all(chunk.chunk_id.startswith("test_doc:") for chunk in chunks)


def test_chunk_by_paragraphs():
    """Test paragraph-based chunking."""
    text = """Paragraph 1. This is the first paragraph.

Paragraph 2. This is the second paragraph with more content.

Paragraph 3. Final paragraph."""
    
    chunks = chunk_by_paragraphs(text, source="doc", max_chunk_size=100)
    assert len(chunks) >= 2


def test_embedding_model():
    """Test embedding generation."""
    model = EmbeddingModel("all-MiniLM-L6-v2")
    
    texts = ["Hello world", "This is a test"]
    embeddings = model.embed(texts)
    
    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32


def test_compute_similarity():
    """Test similarity computation."""
    v1 = np.array([1, 0, 0], dtype=np.float32)
    v2 = np.array([1, 0, 0], dtype=np.float32)
    v3 = np.array([0, 1, 0], dtype=np.float32)
    
    sim_same = compute_similarity(v1, v2)
    sim_ortho = compute_similarity(v1, v3)
    
    assert abs(sim_same - 1.0) < 0.01
    assert abs(sim_ortho) < 0.01


def test_vector_store():
    """Test vector store operations."""
    store = VectorStore(embedding_dim=384)
    
    embeddings = np.random.randn(5, 384).astype(np.float32)
    metadata = [
        {"text": f"chunk {i}", "paper_id": "doc1", "chunk_id": f"chunk_{i}"}
        for i in range(5)
    ]
    
    ids = store.add(embeddings, metadata)
    assert len(ids) == 5
    assert store.__len__() == 5
    
    query_emb = embeddings[0]
    results = store.search(query_emb, k=3)
    assert len(results) == 3
    assert results[0][0] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
