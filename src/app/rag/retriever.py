"""Retrieve chunks from vector store based on queries."""
import numpy as np
from app.llm.schemas import RetrievedChunk
from app.rag.vector_store import VectorStore
from app.rag.embeddings import EmbeddingModel


class RAGRetriever:
    """Retrieve relevant chunks from vector store."""
    
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        """Initialize retriever.
        
        Args:
            vector_store: VectorStore instance
            embedding_model: EmbeddingModel instance
        """
        self.vector_store = vector_store
        embedding_model_obj = embedding_model
        self.embedding_model = embedding_model_obj
    
    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Retrieve top-k chunks for a query.
        
        Args:
            query: Search query text
            k: Number of chunks to retrieve
            
        Returns:
            List of RetrievedChunk objects sorted by relevance
        """
        query_embedding = self.embedding_model.embed_single(query)
        results = self.vector_store.search(query_embedding, k=k)
        
        chunks = []
        for idx, distance, metadata in results:
            score = 1.0 / (1.0 + distance)
            
            chunk = RetrievedChunk(
                paper_id=metadata.get("paper_id", metadata.get("source", f"doc_{idx}")),
                chunk_id=metadata.get("chunk_id", f"chunk_{idx}"),
                text=metadata.get("text", ""),
                title=metadata.get("title", ""),
                authors=metadata.get("authors", []),
                doi=metadata.get("doi"),
                url=metadata.get("url"),
                year=metadata.get("year"),
                page_number=metadata.get("page_number"),
                score=score,
                metadata=metadata.get("metadata", metadata)
            )
            chunks.append(chunk)
        
        return chunks


def create_retriever(vector_store_path: str | None = None, 
                     embedding_model_name: str = "all-MiniLM-L6-v2") -> RAGRetriever:
    """Create a RAG retriever from a vector store.
    
    Args:
        vector_store_path: Path to saved vector store (if None, creates empty store)
        embedding_model_name: Hugging Face model name for embeddings
        
    Returns:
        RAGRetriever instance
    """
    embedding_model = EmbeddingModel(embedding_model_name)
    vector_store = VectorStore(embedding_dim=embedding_model.embedding_dim)
    
    if vector_store_path:
        vector_store.load(vector_store_path)
    
    return RAGRetriever(vector_store, embedding_model)
