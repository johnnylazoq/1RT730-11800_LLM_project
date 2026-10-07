"""LangGraph nodes for the paper correlation pipeline."""

from typing import Any

from paper_corr.graph.schemas import CorrelationResult
from paper_corr.graph.state import GraphState
from paper_corr.llm.prompts import build_pair_analysis_prompt
from paper_corr.rag.retriever import Retriever


async def retrieve_papers_node(state: GraphState, retriever: Retriever) -> dict[str, Any]:
    """Retrieve papers matching query.

    Args:
        state: Graph state.
        retriever: Retriever instance.

    Returns:
        Updated state with papers.
    """
    query = state.get("query", "")
    top_k = state.get("top_k", 10)

    if not query:
        return {"papers": {}}

    # Retrieve papers
    paper_results = retriever.retrieve_papers(query, k=top_k)

    # Build papers dict
    papers = {}
    for paper, score in paper_results:
        papers[paper.paper_id] = {
            "paper": paper,
            "relevance_score": score,
        }

    return {"papers": papers}


async def retrieve_chunks_node(state: GraphState, retriever: Retriever) -> dict[str, Any]:
    """Retrieve chunks from papers.

    Args:
        state: Graph state.
        retriever: Retriever instance.

    Returns:
        Updated state with chunks.
    """
    papers = state.get("papers", {})
    query = state.get("query", "")
    chunks_per_paper = state.get("chunks_per_paper", 5)

    if not papers or not query:
        return {"chunks": []}

    chunks = []
    for paper_id, paper_info in papers.items():
        paper_chunks = retriever.retrieve_chunks(paper_id, query, k=chunks_per_paper)
        chunks.extend(paper_chunks)

    return {"chunks": chunks}


async def build_pairs_node(state: GraphState) -> dict[str, Any]:
    """Build candidate pairs from papers.

    Args:
        state: Graph state.

    Returns:
        Updated state with candidate_pairs.
    """
    papers = state.get("papers", {})
    similarity_threshold = state.get("similarity_threshold", 0.5)
    max_pairs = state.get("max_pairs", 50)

    if len(papers) < 2:
        return {"candidate_pairs": []}

    paper_ids = list(papers.keys())
    pairs = []

    # Create unique pairs (avoid duplicates and self-pairs)
    for i in range(len(paper_ids)):
        for j in range(i + 1, len(paper_ids)):
            paper_a_id = paper_ids[i]
            paper_b_id = paper_ids[j]

            # Pre-filter by relevance score (embedding-based similarity)
            score_a = papers[paper_a_id].get("relevance_score", 0.5)
            score_b = papers[paper_b_id].get("relevance_score", 0.5)
            avg_score = (score_a + score_b) / 2

            if avg_score >= similarity_threshold:
                pairs.append((paper_a_id, paper_b_id))

            if len(pairs) >= max_pairs:
                break

        if len(pairs) >= max_pairs:
            break

    return {"candidate_pairs": pairs}


async def analyze_pair_node(
    state: GraphState,
    paper_a_id: str,
    paper_b_id: str,
    llm_client,
    relation_types: list[str],
) -> dict[str, Any]:
    """Analyze a single paper pair.

    Args:
        state: Graph state.
        paper_a_id: First paper ID.
        paper_b_id: Second paper ID.
        llm_client: LLM client instance.
        relation_types: List of possible relation types.

    Returns:
        Updated state with CorrelationResult.
    """
    papers = state.get("papers", {})
    chunks = state.get("chunks", [])

    if paper_a_id not in papers or paper_b_id not in papers:
        return {"correlations": []}

    paper_a = papers[paper_a_id]["paper"]
    paper_b = papers[paper_b_id]["paper"]

    # Get relevant chunks for each paper
    sections_a = "\n".join(
        [f"[{c.section}] {c.text[:200]}..." for c in chunks if c.paper_id == paper_a_id][:2]
    )
    sections_b = "\n".join(
        [f"[{c.section}] {c.text[:200]}..." for c in chunks if c.paper_id == paper_b_id][:2]
    )

    # Build prompt
    prompt = build_pair_analysis_prompt(
        title_a=paper_a.title,
        abstract_a=paper_a.abstract,
        sections_a=sections_a or "No relevant sections found",
        title_b=paper_b.title,
        abstract_b=paper_b.abstract,
        sections_b=sections_b or "No relevant sections found",
        relation_types=relation_types,
    )

    # Get LLM response
    try:
        response = llm_client.generate(prompt, structured_output=CorrelationResult)

        if isinstance(response, dict):
            result = CorrelationResult(**response)
        else:
            result = response
    except Exception as e:
        # Fallback: create a default result on error
        result = CorrelationResult(
            paper_a_id=paper_a_id,
            paper_b_id=paper_b_id,
            correlation_score=0.5,
            relation_type="unrelated",
            explanation=f"Error analyzing pair: {str(e)}",
            confidence=0.0,
        )

    return {"correlations": [result]}


async def aggregate_node(state: GraphState) -> dict[str, Any]:
    """Aggregate and rank correlations.

    Args:
        state: Graph state.

    Returns:
        Updated state (correlations already aggregated).
    """
    # Correlations are already aggregated via Annotated reducer
    correlations = state.get("correlations", [])

    # Sort by score
    sorted_corr = sorted(correlations, key=lambda x: x.correlation_score, reverse=True)

    return {"correlations": sorted_corr}


async def save_results_node(state: GraphState, output_dir: str = "outputs") -> dict[str, Any]:
    """Save results to timestamped JSON and CSV files.

    Args:
        state: Graph state.
        output_dir: Directory to save results.

    Returns:
        Updated state with output file paths.
    """
    import csv
    import json
    from datetime import datetime
    from pathlib import Path

    correlations = state.get("correlations", [])
    query = state.get("query", "unknown_query")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save JSON
    json_file = output_path / f"correlations_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(
            {
                "query": query,
                "timestamp": timestamp,
                "results": [
                    {
                        "paper_a_id": c.paper_a_id,
                        "paper_b_id": c.paper_b_id,
                        "correlation_score": c.correlation_score,
                        "relation_type": c.relation_type,
                        "evidence": c.evidence,
                        "explanation": c.explanation,
                        "confidence": c.confidence,
                    }
                    for c in correlations
                ],
            },
            f,
            indent=2,
        )

    # Save CSV
    csv_file = output_path / f"correlations_{timestamp}.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["paper_a_id", "paper_b_id", "correlation_score", "relation_type", "explanation", "confidence"]
        )
        for c in correlations:
            writer.writerow(
                [c.paper_a_id, c.paper_b_id, c.correlation_score, c.relation_type, c.explanation, c.confidence]
            )

    return {"output_files": [str(json_file), str(csv_file)]}
