"""Prompt templates for pair analysis."""

from string import Template


PAIR_ANALYSIS_PROMPT = Template("""
Analyze the relationship between two papers and provide a structured assessment.

Paper A:
Title: $title_a
Abstract: $abstract_a
Relevant sections:
$sections_a

Paper B:
Title: $title_b
Abstract: $abstract_b
Relevant sections:
$sections_b

Task:
1. Determine if the papers are correlated (0-1 score).
2. Classify the relationship type: {relation_types}.
3. Identify key evidence (chunks/quotes).
4. Provide a brief explanation.

Respond in JSON format with fields: correlation_score, relation_type, evidence, explanation, confidence.
""")


def build_pair_analysis_prompt(
    title_a: str,
    abstract_a: str,
    sections_a: str,
    title_b: str,
    abstract_b: str,
    sections_b: str,
    relation_types: list[str],
) -> str:
    """Build prompt for analyzing a paper pair.

    Args:
        title_a: Title of paper A.
        abstract_a: Abstract of paper A.
        sections_a: Relevant sections from paper A.
        title_b: Title of paper B.
        abstract_b: Abstract of paper B.
        sections_b: Relevant sections from paper B.
        relation_types: List of possible relation types.

    Returns:
        Formatted prompt string.
    """
    return PAIR_ANALYSIS_PROMPT.substitute(
        title_a=title_a,
        abstract_a=abstract_a,
        sections_a=sections_a,
        title_b=title_b,
        abstract_b=abstract_b,
        sections_b=sections_b,
        relation_types=", ".join(relation_types),
    )
