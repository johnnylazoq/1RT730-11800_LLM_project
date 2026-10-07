"""Build and compile the LangGraph pipeline."""

from typing import Any

from langgraph.graph import StateGraph

from paper_corr.config import load_config
from paper_corr.graph.nodes import (
    aggregate_node,
    analyze_pair_node,
    build_pairs_node,
    retrieve_chunks_node,
    retrieve_papers_node,
    save_results_node,
)
from paper_corr.graph.state import GraphState
from paper_corr.llm.client import LLMClient
from paper_corr.rag.embeddings import EmbeddingModel
from paper_corr.rag.loader import load_papers
from paper_corr.rag.retriever import Retriever
from paper_corr.rag.vectorstore import VectorStore


def build_correlation_graph(config: dict) -> StateGraph:
    """Build the paper correlation pipeline graph.

    Args:
        config: Configuration dictionary.

    Returns:
        Compiled StateGraph.
    """
    graph = StateGraph(GraphState)

    # Get parameters from config
    relation_types = config.get("relation_types", [])
    model = config.get("model", "gpt-4-turbo")
    temperature = config.get("temperature", 0.7)
    embedding_model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    data_path = config.get("data_path", "./data")
    index_path = config.get("index_path", "./data/index")

    # Initialize components
    embedding_model = EmbeddingModel(embedding_model_name)
    vectorstore = VectorStore(embedding_dim=embedding_model.get_embedding_dim())
    papers_dict = {p.paper_id: p for p in load_papers(data_path)}

    # Try to load existing index, otherwise it will be empty
    try:
        vectorstore.load(index_path)
    except FileNotFoundError:
        pass

    retriever = Retriever(vectorstore, embedding_model, papers_dict)
    llm_client = LLMClient(model, temperature=temperature)

    # Add nodes
    graph.add_node("retrieve_papers", lambda state: retrieve_papers_node(state, retriever))
    graph.add_node("retrieve_chunks", lambda state: retrieve_chunks_node(state, retriever))
    graph.add_node("build_pairs", build_pairs_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("save_results", save_results_node)

    # Add analyze_pair as a dynamic node
    def analyze_pair_wrapper(state: GraphState) -> dict:
        """Wrap analyze_pair for all candidate pairs."""
        candidate_pairs = state.get("candidate_pairs", [])
        correlations = []

        for paper_a_id, paper_b_id in candidate_pairs:
            result = analyze_pair_node(state, paper_a_id, paper_b_id, llm_client, relation_types)
            correlations.extend(result.get("correlations", []))

        return {"correlations": correlations}

    graph.add_node("analyze_pairs", analyze_pair_wrapper)

    # Add edges
    graph.add_edge("START", "retrieve_papers")
    graph.add_edge("retrieve_papers", "retrieve_chunks")
    graph.add_edge("retrieve_chunks", "build_pairs")
    graph.add_edge("build_pairs", "analyze_pairs")
    graph.add_edge("analyze_pairs", "aggregate")
    graph.add_edge("aggregate", "save_results")
    graph.add_edge("save_results", "END")

    return graph.compile()


def run_pipeline(query: str = "", config_path: str = "configs/default.yaml", seed_paper_ids: list[str] | None = None) -> dict:
    """Run the correlation pipeline.

    Args:
        query: User query or search string.
        config_path: Path to configuration file.
        seed_paper_ids: Optional initial paper IDs to include.

    Returns:
        Results dictionary with correlations and metadata.
    """
    # Load config
    config = load_config(config_path)

    # Build graph
    graph = build_correlation_graph(config)

    # Prepare initial state
    initial_state: GraphState = {
        "query": query,
        "seed_paper_ids": seed_paper_ids or [],
        "papers": {},
        "chunks": [],
        "candidate_pairs": [],
        "correlations": [],
        "errors": [],
        # Add config parameters to state
        "top_k": config.get("top_k", 10),
        "chunks_per_paper": config.get("chunks_per_paper", 5),
        "max_pairs": config.get("max_pairs", 50),
        "similarity_threshold": config.get("similarity_threshold", 0.5),
    }

    # Run graph
    result = graph.invoke(initial_state)

    return result
