"""LangGraph workflow for claim verification."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from langgraph.graph import StateGraph, START, END
from app.graph.state import GraphState
from app.graph.nodes.group_papers import group_chunks
from app.graph.nodes.analyze_paper import analyze_single_paper
from app.graph.nodes.aggregate_evidence import aggregate_evidence
from app.graph.nodes.rag import retrieve_evidence
from app.llm.schemas import RetrievedChunk, PaperAnalysis, PaperAssessment


def build_graph(*, llm=None, max_workers=1, retries=1, retriever=None, **group_options):
    """Build LangGraph workflow for claim verification.
    
    Args:
        llm: LLM client (optional, uses default if None)
        max_workers: Number of parallel workers for analysis
        retries: Number of retries for failed analyses
        retriever: RAGRetriever instance (optional, for RAG-based retrieval)
        **group_options: Additional options for grouping
        
    Returns:
        Compiled LangGraph workflow
    """
    if max_workers < 1 or retries < 0:
        raise ValueError("Invalid worker or retry count")

    def group(state):
        if not state["claim"].strip():
            raise ValueError("Claim must not be blank")
        records = [RetrievedChunk.model_validate(r) for r in state["retrieved_chunks"]]
        papers, notes = group_chunks(records, **group_options)
        return {"papers": papers, "limitations": notes}

    def analyze(state):
        def run(paper):
            error = None
            for attempt in range(retries + 1):
                try:
                    return analyze_single_paper(state["claim"], paper, llm=llm, retry_feedback=error)
                except Exception as exc:
                    error = type(exc).__name__
            return PaperAnalysis(paper_id=paper.paper_id, title=paper.title, status="failed",
                error=error, assessment=PaperAssessment(stance="insufficient",
                    explanation="Paper could not be analyzed due to a processing failure."))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # Preserve graph callbacks and tracing parents in each worker.
            futures = [pool.submit(copy_context().run, run, paper) for paper in state["papers"]]
            return {"analyses": [future.result() for future in futures]}

    def aggregate(state):
        feedback = None
        for attempt in range(retries + 1):
            try:
                return {"final": aggregate_evidence(state["claim"], state["papers"],
                    state["analyses"], limitations=state["limitations"],
                    evidence_cutoff=state.get("evidence_cutoff"), llm=llm, retry_feedback=feedback)}
            except Exception as exc:
                feedback = str(exc)
                if attempt == retries:
                    raise

    builder = StateGraph(GraphState)
    
    if retriever:
        def rag_retrieve(state):
            return retrieve_evidence(state, retriever)
        builder.add_node("retrieve_rag", rag_retrieve)
    
    builder.add_node("group_papers", group)
    builder.add_node("analyze_papers", analyze)
    builder.add_node("aggregate_evidence", aggregate)
    
    if retriever:
        builder.add_edge(START, "retrieve_rag")
        builder.add_edge("retrieve_rag", "group_papers")
    else:
        builder.add_edge(START, "group_papers")
    
    builder.add_edge("group_papers", "analyze_papers")
    builder.add_edge("analyze_papers", "aggregate_evidence")
    builder.add_edge("aggregate_evidence", END)
    return builder.compile()


def verify_claim(claim, retrieved_chunks, *, evidence_cutoff=None, retriever=None, **options):
    """Verify a claim against retrieved evidence.
    
    Args:
        claim: The claim to verify
        retrieved_chunks: List of RetrievedChunk objects or dicts
        evidence_cutoff: Optional cutoff date for evidence
        retriever: Optional RAGRetriever for dynamic retrieval
        **options: Additional options (max_workers, retries, etc.)
        
    Returns:
        FinalAssessment with verdict, confidence, and sources
    """
    return build_graph(retriever=retriever, **options).invoke({
        "claim": claim, 
        "retrieved_chunks": retrieved_chunks,
        "evidence_cutoff": evidence_cutoff
    })["final"]
