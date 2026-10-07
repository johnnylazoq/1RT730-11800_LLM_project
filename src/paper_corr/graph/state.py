"""LangGraph state definition."""

from typing import Annotated, Any

from typing_extensions import TypedDict

from paper_corr.graph.schemas import CorrelationResult


class GraphState(TypedDict):
    """State passed through the LangGraph pipeline."""

    query: str  # User query
    seed_paper_ids: list[str]  # Optional initial paper IDs
    papers: dict[str, Any]  # paper_id -> Paper
    chunks: list[Any]  # List of Chunk objects
    candidate_pairs: list[tuple[str, str]]  # Pairs to analyze
    correlations: Annotated[
        list[CorrelationResult],
        lambda x, y: y if isinstance(y, list) else x + (y if isinstance(y, list) else [y]),
    ]  # Reducer: merge correlations
    errors: list[dict]  # Error log: {node, error, details}
