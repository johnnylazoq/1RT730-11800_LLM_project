
from typing import Annotated, Any, Literal
from pydantic import BaseModel, Field



Stance = Literal[
    "supports",
    "contradicts",
    "mixed",
    "neutral",
    "insufficient",
]

Confidence = Literal["low", "moderate", "high"]



class EvidenceChunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    page_number: int | None = None
    score: float | None = Field(default=None, allow_inf_nan=False)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(EvidenceChunk):
    title: str
    authors: list[str] = Field(default_factory=list)
    doi: str | None = None
    url: str | None = None
    year: int | None = None


class ScientificPaper(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    doi: str | None = None
    url: str | None = None
    year: int | None = None
    chunks: list[EvidenceChunk] = Field(default_factory=list)



class EvidenceFinding(BaseModel):
    statement: str = Field(min_length=1, max_length=280)
    chunk_id: str


class PaperAssessment(BaseModel):
    stance: Stance
    explanation: str = Field(min_length=1, max_length=240, description="One short sentence stating the conclusion.")
    findings: list[EvidenceFinding] = Field(default_factory=list, max_length=3)
    limitations: list[Annotated[str, Field(max_length=200)]] = Field(default_factory=list, max_length=2)
    study_design: str = Field(default="unknown", max_length=80)
    quality_considerations: list[Annotated[str, Field(max_length=200)]] = Field(default_factory=list, max_length=1)


class PaperAnalysis(BaseModel):
    paper_id: str
    title: str
    assessment: PaperAssessment
    status: Literal["ok", "failed"] = "ok"
    error: str | None = None



class EvidenceSummaryItem(BaseModel):
    statement: str
    citation_ids: list[str] = Field(min_length=1, description="Required nonempty list of exact allowed citation IDs, such as [\"1\"].")


class EvidenceEntry(BaseModel):
    paper_id: str
    stance: Stance
    summary: str
    study_design: str = "unknown"
    status: Literal["ok", "failed"] = "ok"


class SourceReference(BaseModel):
    citation_id: str
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    doi: str | None = None
    url: str | None = None
    chunk_id: str
    page_number: int | None = None
    year: int | None = None
    excerpt: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AggregationAssessment(BaseModel):
    verdict: Literal["supports", "contradicts", "mixed", "insufficient"]
    confidence: Confidence
    evidence_summary: str = Field(min_length=1, max_length=1200,
        description="One paragraph of 2–3 short sentences synthesizing all paper analyses.")
    key_findings: list[EvidenceSummaryItem] = Field(min_length=2, max_length=3,
        description="2–3 distinct scientific results, each citing supplied chunk citation IDs.")
    limitations: list[str] = Field(min_length=2, max_length=3,
        description="2–3 concise, distinct reasons for uncertainty, merging study and pipeline limitations.")


class FinalAssessment(BaseModel):
    """Internal assessment; public_result exposes the six user-facing fields."""
    claim: str
    verdict: Literal["supports", "contradicts", "mixed", "insufficient"]
    confidence: Confidence
    evidence_summary: str
    key_findings: list[EvidenceSummaryItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    analyses: list[PaperAnalysis] = Field(default_factory=list)
    evidence_cutoff: str | None = None

    def public_result(self) -> dict:
        sources = []
        seen = set()
        cited = {citation for finding in self.key_findings for citation in finding.citation_ids}
        for source in self.sources:
            if source.citation_id not in cited:
                continue
            if source.paper_id not in seen:
                seen.add(source.paper_id)
                sources.append({"title": source.title,
                    "url": source.url or (f"https://doi.org/{source.doi}" if source.doi else None)})
        return {
            "verdict": {"supports": "Supports", "contradicts": "Contradicts",
                        "mixed": "Mixed", "insufficient": "Insufficient Evidence"}[self.verdict],
            "confidence": self.confidence.title(),
            "evidence_summary": self.evidence_summary,
            "key_findings": [finding.statement for finding in self.key_findings],
            "limitations": self.limitations,
            "sources": sources,
        }
