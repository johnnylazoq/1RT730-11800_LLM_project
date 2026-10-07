"""Tests for LangGraph pipeline."""

import pytest

from paper_corr.graph.build_graph import build_correlation_graph
from paper_corr.graph.state import GraphState


@pytest.fixture
def sample_config():
    """Sample configuration."""
    return {
        "top_k": 10,
        "chunks_per_paper": 5,
        "max_pairs": 50,
        "similarity_threshold": 0.5,
        "relation_types": ["same_problem", "builds_on", "contradicts"],
        "model": "gpt-4-turbo",
        "temperature": 0.7,
    }


@pytest.mark.skip(reason="Graph stub has no edges yet - will pass after node implementation")
def test_graph_builds(sample_config):
    """Test that the graph builds without errors.

    Note: This test will pass once nodes and edges are implemented in build_graph.py.
    """
    graph = build_correlation_graph(sample_config)
    assert graph is not None


def test_initial_state():
    """Test GraphState initialization."""
    state: GraphState = {
        "query": "machine learning optimization",
        "seed_paper_ids": [],
        "papers": {},
        "chunks": [],
        "candidate_pairs": [],
        "correlations": [],
        "errors": [],
    }
    assert state["query"] == "machine learning optimization"
    assert isinstance(state["correlations"], list)
