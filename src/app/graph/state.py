from typing import TypedDict
from app.llm.schemas import RetrievedChunk, ScientificPaper, PaperAnalysis, FinalAssessment


class GraphState(TypedDict, total=False):
    claim: str
    retrieved_chunks: list[RetrievedChunk]
    papers: list[ScientificPaper]
    analyses: list[PaperAnalysis]
    limitations: list[str]
    evidence_cutoff: str | None
    final: FinalAssessment
