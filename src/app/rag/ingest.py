"""Orchestrate the complete RAG pipeline: collect, extract, chunk, embed, store."""
import json
from pathlib import Path
from app.rag.collect_papers import fetch_papers, resolve_paper
from app.rag.extract_documents import extract_pages_from_pdf
from app.rag.chunk_documents import chunk_pages, TextChunk
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore
from app.llm.schemas import RetrievedChunk


def ingest_papers_from_api(query: str, 
                          vector_store_path: str = "vector_db",
                          embedding_model_name: str = "all-MiniLM-L6-v2",
                          limit: int = 10,
                          chunk_strategy: str = "paragraph",
                          abstract_fallback: bool = True) -> VectorStore:
    """Ingest papers from Semantic Scholar API into vector store.
    
    Args:
        query: Search query for papers
        vector_store_path: Path to save vector store
        embedding_model_name: Embedding model name
        limit: Max papers to fetch
        
    Returns:
        Populated VectorStore
    """
    print(f"Fetching papers for: {query}")
    papers = fetch_papers(query, limit=limit)
    chunks, reports = [], []
    for paper in papers:
        evidence, report = resolve_paper(paper, strategy=chunk_strategy,
                                         abstract_fallback=abstract_fallback)
        chunks.extend(evidence)
        reports.append(report)
        print(f"{report['paper_id']}: {report['full_text_status']} ({len(evidence)} chunks)")
    destination = Path(vector_store_path)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "ingestion_report.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8")
    return ingest_chunks(chunks, vector_store_path, embedding_model_name)


def ingest_pdfs(pdf_dir: str,
               vector_store_path: str = "vector_db",
               embedding_model_name: str = "all-MiniLM-L6-v2",
               chunk_strategy: str = "paragraph") -> VectorStore:
    """Ingest PDFs into vector store.
    
    Args:
        pdf_dir: Directory containing PDFs
        vector_store_path: Path to save vector store
        embedding_model_name: Embedding model name
        chunk_strategy: "paragraph" or "page"
        
    Returns:
        Populated VectorStore
    """
    if chunk_strategy not in {"paragraph", "page"}:
        raise ValueError("chunk_strategy must be 'paragraph' or 'page'")
    directory = Path(pdf_dir)
    if not directory.is_dir():
        raise NotADirectoryError(pdf_dir)
    text_chunks = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() != ".pdf":
            continue
        pages = extract_pages_from_pdf(path)
        text_chunks.extend(chunk_pages(pages, path.name, chunk_strategy))
    return ingest_chunks(_text_chunks_to_retrieved(text_chunks),
                         vector_store_path, embedding_model_name)


def ingest_chunks(chunks: list[RetrievedChunk],
                 vector_store_path: str = "vector_db",
                 embedding_model_name: str = "all-MiniLM-L6-v2", *,
                 embedding_model: EmbeddingModel | None = None) -> VectorStore:
    """Ingest RetrievedChunk objects into vector store.
    
    Args:
        chunks: List of RetrievedChunk objects
        vector_store_path: Path to save vector store
        embedding_model_name: Embedding model name
        
    Returns:
        Populated VectorStore
    """
    if not chunks:
        raise ValueError("No chunks to ingest")
    
    print(f"Ingesting {len(chunks)} chunks")
    
    if embedding_model is None:
        embedding_model = EmbeddingModel(embedding_model_name)
    vector_store = VectorStore(embedding_dim=embedding_model.embedding_dim)
    
    texts = [chunk.text for chunk in chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = embedding_model.embed(texts)
    
    metadata_list = [
        {
            "paper_id": chunk.paper_id,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "title": chunk.title,
            "authors": chunk.authors,
            "doi": chunk.doi,
            "url": chunk.url,
            "year": chunk.year,
            "page_number": chunk.page_number,
            "score": chunk.score,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]
    
    print(f"Adding to vector store...")
    vector_store.add(embeddings, metadata_list)
    
    print(f"Saving to {vector_store_path}")
    vector_store.save(vector_store_path)
    
    return vector_store


def _text_chunks_to_retrieved(text_chunks: list[TextChunk]) -> list[RetrievedChunk]:
    """Convert TextChunk to RetrievedChunk format."""
    chunks = []
    for tc in text_chunks:
        chunk = RetrievedChunk(
            paper_id=tc.source,
            chunk_id=tc.chunk_id,
            text=tc.text,
            title=tc.source,
            authors=[],
            doi=None,
            url=None,
            year=None,
            page_number=tc.page_number,
            score=None,
            metadata={
                "evidence_type": "pdf_chunk",
                "provider": "local_pdf",
                "paragraph_number": tc.paragraph_number
            }
        )
        chunks.append(chunk)
    return chunks
