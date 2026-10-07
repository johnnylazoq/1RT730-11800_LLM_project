"""Tests for schemas and data structures."""

import pytest
from pydantic import ValidationError

from paper_corr.graph.schemas import CorrelationMatrix, CorrelationResult


def test_correlation_result_valid():
    """Test valid CorrelationResult."""
    result = CorrelationResult(
        paper_a_id="paper_1",
        paper_b_id="paper_2",
        correlation_score=0.85,
        relation_type="builds_on",
        explanation="Paper 2 builds on the methods from Paper 1",
    )
    assert result.correlation_score == 0.85
    assert result.relation_type == "builds_on"


def test_correlation_result_invalid_score():
    """Test CorrelationResult with invalid score."""
    with pytest.raises(ValidationError):
        CorrelationResult(
            paper_a_id="paper_1",
            paper_b_id="paper_2",
            correlation_score=1.5,  # Out of range
            relation_type="builds_on",
            explanation="Test",
        )


def test_correlation_matrix_deduplication():
    """Test that CorrelationMatrix deduplicates pairs."""
    matrix = CorrelationMatrix()

    result_ab = CorrelationResult(
        paper_a_id="paper_1",
        paper_b_id="paper_2",
        correlation_score=0.8,
        relation_type="builds_on",
        explanation="Test",
    )

    result_ba = CorrelationResult(
        paper_a_id="paper_2",
        paper_b_id="paper_1",
        correlation_score=0.9,
        relation_type="same_problem",
        explanation="Updated",
    )

    matrix.add(result_ab)
    matrix.add(result_ba)

    # Should deduplicate: only one entry for (paper_1, paper_2)
    assert len(matrix.correlations) == 1


def test_correlation_matrix_sorting():
    """Test that correlations are sorted by score."""
    matrix = CorrelationMatrix()

    result1 = CorrelationResult(
        paper_a_id="paper_1",
        paper_b_id="paper_2",
        correlation_score=0.5,
        relation_type="unrelated",
        explanation="Low correlation",
    )

    result2 = CorrelationResult(
        paper_a_id="paper_1",
        paper_b_id="paper_3",
        correlation_score=0.9,
        relation_type="builds_on",
        explanation="High correlation",
    )

    matrix.add(result1)
    matrix.add(result2)

    sorted_results = matrix.get_sorted(reverse=True)
    assert sorted_results[0].correlation_score == 0.9
    assert sorted_results[1].correlation_score == 0.5
