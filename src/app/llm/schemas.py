
from typing import Any, Literal
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
    statement: str
    chunk_id: str


class PaperAssessment(BaseModel):
    stance: Stance
    explanation: str
    findings: list[EvidenceFinding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    study_design: str = "unknown"
    quality_considerations: list[str] = Field(default_factory=list)


class PaperAnalysis(BaseModel):
    paper_id: str
    title: str
    assessment: PaperAssessment
    status: Literal["ok", "failed"] = "ok"
    error: str | None = None



class EvidenceSummaryItem(BaseModel):
    statement: str
    citation_ids: list[str] = Field(default_factory=list)


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
    verdict: Stance
    confidence: Confidence
    confidence_reason: str
    evidence_summary: list[EvidenceSummaryItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)




# ---------- Final Output ----------

class FinalAssessment(BaseModel):
    claim: str
    verdict: Stance
    confidence: Confidence
    confidence_reason: str

    evidence_summary: list[EvidenceSummaryItem] = Field(
        default_factory=list
    )
    evidence: list[EvidenceEntry] = Field(
        default_factory=list
    )
    limitations: list[str] = Field(
        default_factory=list
    )
    sources: list[SourceReference] = Field(
        default_factory=list
    )
    analyses: list[PaperAnalysis] = Field(default_factory=list)
    evidence_cutoff: str | None = None
