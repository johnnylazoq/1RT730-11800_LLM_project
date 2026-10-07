"""Build vector index from papers in data/."""

import sys
from pathlib import Path

# Add src/ to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from paper_corr.config import load_config
from paper_corr.rag.chunker import chunk_papers
from paper_corr.rag.embeddings import EmbeddingModel
from paper_corr.rag.loader import load_papers
from paper_corr.rag.vectorstore import VectorStore


def main():
    """Build and save vector index."""
    config = load_config()
    data_path = config.get("data_path", "./data")
    index_path = config.get("index_path", "./data/index")
    embedding_model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size = config.get("chunk_size", 512)
    chunk_overlap = config.get("chunk_overlap", 64)

    print(f"Loading papers from {data_path}...")
    papers = load_papers(data_path)
    print(f"✓ Loaded {len(papers)} papers")

    if not papers:
        print("No papers found. Create JSON files in data/raw/ or data/processed/")
        return

    print(f"Chunking papers (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    chunks = chunk_papers(papers, chunk_size=chunk_size, overlap=chunk_overlap)
    print(f"✓ Created {len(chunks)} chunks")

    print(f"Embedding chunks with {embedding_model_name}...")
    embedding_model = EmbeddingModel(embedding_model_name)
    embeddings = embedding_model.embed_batch([c.text for c in chunks])
    print(f"✓ Embedded {len(embeddings)} chunks")

    print(f"Building FAISS index...")
    vectorstore = VectorStore(embedding_dim=embedding_model.get_embedding_dim())
    vectorstore.add_chunks(chunks, embeddings)
    print(f"✓ Index created with {len(chunks)} chunks")

    print(f"Saving index to {index_path}...")
    vectorstore.save(index_path)
    print(f"✓ Index saved!")

    print("\n✅ Vector index built successfully!")
    print(f"Ready to run: python scripts/run_graph.py --query 'your query'")


if __name__ == "__main__":
    main()
