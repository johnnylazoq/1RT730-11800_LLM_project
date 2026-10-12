from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Paper:
    title: str
    authors: str
    year: int
    url: str
    abstract: str
    tags: tuple[str, ...]
    relevance_note: str
    stance: str
    study_type: str
    page: int
    section: str
    snippet: str


STARTER_PAPERS: tuple[Paper, ...] = (
    Paper(
        title="Rumor Detection on Social Media: Datasets, Methods and Opportunities",
        authors="Quanzhi Li, Qiong Zhang, Luo Si, Yingchi Liu",
        year=2019,
        url="https://arxiv.org/abs/1911.07199",
        abstract=(
            "Surveys rumor detection work on social media, including datasets, "
            "content features, social context, propagation signals, and future "
            "research opportunities."
        ),
        tags=("rumor", "social media", "datasets", "methods", "misinformation"),
        relevance_note="It explains which signals rumor-detection systems can use.",
        stance="Background",
        study_type="Survey",
        page=1,
        section="Overview",
        snippet="Surveys datasets, features, social context, and propagation signals for rumor detection.",
    ),
    Paper(
        title="SemEval-2019 Task 7: RumourEval, Determining Rumour Veracity and Support for Rumours",
        authors="Genevieve Gorrell, Elena Kochkina, Maria Liakata, Ahmet Aker, Arkaitz Zubiaga, Kalina Bontcheva, Leon Derczynski",
        year=2019,
        url="https://aclanthology.org/S19-2147/",
        abstract=(
            "Presents a shared task for rumor stance prediction and rumor "
            "verification using social media conversations from Twitter and Reddit."
        ),
        tags=("rumor", "veracity", "stance", "social media", "dataset"),
        relevance_note="It connects rumor claims to stance and veracity labels.",
        stance="Background",
        study_type="Shared task",
        page=1,
        section="Task description",
        snippet="Defines rumor stance and veracity prediction tasks for social media conversations.",
    ),
    Paper(
        title="FEVER: a Large-scale Dataset for Fact Extraction and VERification",
        authors="James Thorne, Andreas Vlachos, Christos Christodoulopoulos, Arpit Mittal",
        year=2018,
        url="https://aclanthology.org/N18-1074/",
        abstract=(
            "Introduces a large claim verification dataset where claims are "
            "classified as supported, refuted, or not enough information, with "
            "evidence sentences recorded for supported and refuted claims."
        ),
        tags=("fact checking", "claim verification", "evidence", "dataset", "fever"),
        relevance_note="It frames the final answer as support, refute, or not enough information.",
        stance="Background",
        study_type="Dataset",
        page=1,
        section="Dataset construction",
        snippet="Defines supported, refuted, and not-enough-information labels for claim verification.",
    ),
    Paper(
        title="A Survey on Automated Fact-Checking",
        authors="Zhijiang Guo, Michael Schlichtkrull, Andreas Vlachos",
        year=2022,
        url="https://aclanthology.org/2022.tacl-1.11/",
        abstract=(
            "Surveys automated fact-checking research in NLP, including claim "
            "detection, evidence retrieval, claim verification, datasets, and "
            "open research challenges."
        ),
        tags=("fact checking", "claim detection", "evidence retrieval", "verification", "survey"),
        relevance_note="It explains the overall claim-checking pipeline.",
        stance="Background",
        study_type="Survey",
        page=1,
        section="Automated fact-checking pipeline",
        snippet="Surveys claim detection, evidence retrieval, and claim verification methods.",
    ),
    Paper(
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        authors="Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuettler, Mike Lewis, Wen-tau Yih, Tim Rocktaschel, Sebastian Riedel, Douwe Kiela",
        year=2020,
        url="https://arxiv.org/abs/2005.11401",
        abstract=(
            "Introduces retrieval-augmented generation, combining a parametric "
            "sequence-to-sequence model with a non-parametric memory of retrieved "
            "passages for knowledge-intensive generation tasks."
        ),
        tags=("rag", "retrieval", "generation", "factuality", "llm"),
        relevance_note="It justifies retrieval before generation in the app design.",
        stance="Background",
        study_type="Method paper",
        page=1,
        section="Retrieval-augmented generation",
        snippet="Combines retrieved passages with generation for knowledge-intensive NLP tasks.",
    ),
)


def _tokenize(text: str) -> set[str]:
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    tokens: set[str] = set()
    for token in cleaned.split():
        if len(token) <= 2:
            continue
        tokens.add(token)
        if token.endswith("s") and len(token) > 4:
            tokens.add(token[:-1])
    return tokens


def _score_paper(claim_tokens: set[str], paper: Paper) -> float:
    paper_tokens = _tokenize(" ".join((paper.title, paper.abstract, " ".join(paper.tags))))
    overlap = len(claim_tokens & paper_tokens)
    tag_bonus = sum(0.4 for tag in paper.tags if any(part in claim_tokens for part in tag.split()))
    if overlap == 0 and tag_bonus == 0:
        return 0.0
    recency_bonus = max(0, paper.year - 2021) * 0.08
    return overlap + tag_bonus + recency_bonus


def retrieve_papers(claim: str, top_k: int) -> list[dict[str, Any]]:
    """Starter paper retrieval.

    Replace this with the scraper, PDF parser, or vector database once the RAG
    part is ready. Keep the returned keys stable so the UI does not need to move.
    """
    claim_tokens = _tokenize(claim)
    scored = sorted(
        ((_score_paper(claim_tokens, paper), paper) for paper in STARTER_PAPERS),
        key=lambda item: item[0],
        reverse=True,
    )

    best_score = max((score for score, _ in scored), default=1.0) or 1.0
    sources: list[dict[str, Any]] = []
    for rank, (score, paper) in enumerate(scored[:top_k], start=1):
        if score <= 0:
            continue
        relevance = round(min(1.0, score / best_score), 2)
        sources.append(
            {
                "rank": rank,
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.year,
                "url": paper.url,
                "relevance": relevance,
                "why_selected": _selection_reason(claim_tokens, paper),
                "abstract": paper.abstract,
                "stance": paper.stance,
                "study_type": paper.study_type,
                "page": paper.page,
                "section": paper.section,
                "snippet": paper.snippet,
            }
        )
    return sources


def _selection_reason(claim_tokens: set[str], paper: Paper) -> str:
    matched_tags = [tag for tag in paper.tags if any(part in claim_tokens for part in tag.split())]
    if matched_tags:
        return (
            f"{paper.relevance_note} Matched claim theme: {', '.join(matched_tags[:3])}."
        )
    return paper.relevance_note


def generate_answer(claim: str, sources: list[dict[str, Any]], model_name: str) -> str:
    """Starter answer generator.

    Replace this with the final LLM call. The UI expects Markdown text.
    """
    if not sources:
        return (
            "I could not retrieve enough evidence to make a grounded assessment. "
            "Try a more specific rumor or add papers to the retrieval index."
        )

    strongest = sources[0]
    source_titles = ", ".join(f"[{source['rank']}]" for source in sources[:3])
    return (
        f"**Preliminary assessment:** the retrieved papers are relevant background, "
        f"but they do not directly verify the specific rumor.\n\n"
        f"The strongest retrieved source is **{strongest['title']}**. "
        f"Reason: {strongest['why_selected']} In the full system, the top "
        f"{len(sources)} evidence chunks would be passed to `{model_name}` to compare support, "
        f"contradiction, uncertainty, and source quality.\n\n"
        f"At this stage, the system should not claim support or refutation. It can only say "
        f"that relevant background exists and that claim-specific evidence is still missing. "
        f"See sources {source_titles}."
    )


def build_evidence_summary(sources: list[dict[str, Any]]) -> list[str]:
    if not sources:
        return ["No relevant evidence was retrieved for this claim."]

    top_titles = ", ".join(source["title"] for source in sources[:2])
    return [
        f"Retrieved background evidence from {len(sources)} paper(s), led by {top_titles}.",
        "The current starter corpus explains methods and datasets, but does not yet contain claim-specific study findings.",
    ]


def build_limitations(sources: list[dict[str, Any]]) -> list[str]:
    limitations = [
        "Starter corpus contains paper-level metadata and summaries, not full PDF chunks yet.",
        "Current retrieval is keyword-based; embeddings or a vector index should replace it.",
        "Verdict is conservative until claim-specific evidence chunks are retrieved.",
    ]
    if not sources:
        limitations.insert(0, "No source passed the retrieval threshold.")
    return limitations


def choose_confidence(sources: list[dict[str, Any]]) -> dict[str, str]:
    if not sources:
        return {
            "label": "Low",
            "rationale": "No relevant evidence was retrieved.",
        }
    return {
        "label": "Low",
        "rationale": "Sources are relevant background, but not direct evidence for the specific claim.",
    }


def choose_verdict(sources: list[dict[str, Any]]) -> dict[str, str]:
    """Return a conservative verdict for the starter corpus.

    The current corpus contains research papers about methods, not live evidence
    about the user's specific claim. Until the final verifier is connected, the
    honest verdict is therefore "Not enough evidence".
    """
    if not sources:
        return {
            "label": "Not enough evidence",
            "tone": "caution",
            "rationale": "No relevant evidence was retrieved for this claim.",
        }

    return {
        "label": "Not enough evidence",
        "tone": "caution",
        "rationale": (
            "Retrieved papers are relevant background, but they do not directly "
            "prove or disprove the specific rumor."
        ),
    }


def run_pipeline(claim: str, top_k: int = 5, model_name: str = "starter-rag-baseline") -> dict[str, Any]:
    """Run the end-to-end Rumor2Reality pipeline.

    This starter version gives the UI a realistic response shape. Your group can
    replace each stage while preserving the final dictionary keys.
    """
    started_at = time.perf_counter()
    timings: dict[str, float] = {}

    stage_start = time.perf_counter()
    time.sleep(0.2)
    scraped_count = len(STARTER_PAPERS)
    timings["scrape_or_load_papers"] = round(time.perf_counter() - stage_start, 3)

    stage_start = time.perf_counter()
    sources = retrieve_papers(claim, top_k=top_k)
    time.sleep(0.2)
    timings["rag_retrieval"] = round(time.perf_counter() - stage_start, 3)

    stage_start = time.perf_counter()
    answer = generate_answer(claim, sources, model_name=model_name)
    verdict = choose_verdict(sources)
    confidence = choose_confidence(sources)
    evidence_summary = build_evidence_summary(sources)
    limitations = build_limitations(sources)
    time.sleep(0.2)
    timings["llm_generation"] = round(time.perf_counter() - stage_start, 3)

    timings["total"] = round(time.perf_counter() - started_at, 3)
    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": confidence,
        "evidence_summary": evidence_summary,
        "limitations": limitations,
        "answer": answer,
        "sources": sources,
        "timings": timings,
        "metadata": {
            "model": model_name,
            "papers_available": scraped_count,
            "top_k": top_k,
            "pipeline_mode": "starter-paper-corpus",
        },
    }
