"""Pydantic schemas for graph data structures."""

from typing import Annotated, Any

from pydantic import BaseModel, Field

from paper_corr.rag.chunker import Chunk
from paper_corr.rag.loader import Paper


class CorrelationResult(BaseModel):
    """Result of analyzing a pair of papers."""

    paper_a_id: str = Field(..., description="ID of first paper")
    paper_b_id: str = Field(..., description="ID of second paper")
    correlation_score: float = Field(..., ge=0, le=1, description="Correlation score 0-1")
    relation_type: str = Field(..., description="Type of relation (e.g., same_problem, builds_on)")
    evidence: list[str] = Field(default_factory=list, description="Chunk IDs or quotes supporting the correlation")
    explanation: str = Field(..., description="Brief explanation of the correlation")
    confidence: float = Field(default=0.5, ge=0, le=1, description="Confidence in the analysis")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class CorrelationMatrix:
    """Store correlations with deduplication and ranking."""

    def __init__(self):
        """Initialize empty matrix."""
        self.correlations: dict[tuple[str, str], CorrelationResult] = {}

    def add(self, result: CorrelationResult):
        """Add a correlation result, deduplicating pairs.

        Args:
            result: CorrelationResult to add.
        """
        key = tuple(sorted([result.paper_a_id, result.paper_b_id]))
        self.correlations[key] = result

    def get_sorted(self, reverse: bool = True) -> list[CorrelationResult]:
        """Get correlations sorted by score.

        Args:
            reverse: Sort descending (highest first).

        Returns:
            List of CorrelationResult sorted by score.
        """
        return sorted(
            self.correlations.values(),
            key=lambda x: x.correlation_score,
            reverse=reverse,
        )
