"""LangGraph node for RAG retrieval."""
from app.rag.retriever import RAGRetriever
from app.graph.state import GraphState


def retrieve_evidence(state: GraphState, retriever: RAGRetriever | None = None) -> dict:
    """Retrieve evidence chunks for a claim from RAG vector store.
    
    This node can be inserted into the workflow to add RAG-based retrieval.
    If retriever is None, uses passed RetrievedChunks from state.
    
    Args:
        state: Current graph state
        retriever: RAGRetriever instance (optional)
        
    Returns:
        Updated state with retrieved_chunks
    """
    claim = state.get("claim", "")
    
    if not claim.strip():
        return {"retrieved_chunks": []}
    
    if retriever is None:
        return {"retrieved_chunks": state.get("retrieved_chunks", [])}
    
    print(f"Retrieving evidence for: {claim}")
    chunks = retriever.retrieve(claim, k=4)
    
    return {"retrieved_chunks": [chunk.model_dump() for chunk in chunks]}
