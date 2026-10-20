from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

# Support launching this file's sibling app.py from any working directory.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from app.graph.workflow import verify_claim
from app.llm.client import get_llm

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "frob/qwen3.5-instruct:9b")


def pipeline_events(claim, top_k=4, model_name=DEFAULT_MODEL,
                    evidence_source="Semantic Scholar", vector_store_path="", *, llm=None):
    """Yield actual stage transitions followed by the UI result contract."""
    claim = (claim or "").strip()
    if not claim:
        raise ValueError("Enter a claim to check.")
    top_k = int(top_k)
    if not 1 <= top_k <= 20:
        raise ValueError("Evidence count must be between 1 and 20.")
    if not model_name.strip():
        raise ValueError("Enter an Ollama model name.")
    if evidence_source not in ("Semantic Scholar", "Existing vector store"):
        raise ValueError("Choose a supported evidence source.")
    started = perf_counter()
    timings = {}
    yield "scrape", None
    from app.rag.retriever import create_retriever
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    if evidence_source == "Semantic Scholar":
        from app.rag.collect_papers import fetch_papers, papers_to_chunks
        from app.rag.ingest import ingest_chunks
        chunks = papers_to_chunks(fetch_papers(claim, limit=min(100, max(20, top_k * 3))))
        available = len(chunks)
        timings["load_seconds"] = perf_counter() - started
        yield "retrieve", None
        if chunks:
            # Each request owns its index so concurrent claims cannot overwrite it.
            with TemporaryDirectory(prefix="rumor2reality-") as directory:
                ingest_chunks(chunks, directory, embedding_model)
                selected = create_retriever(directory, embedding_model).retrieve(claim, k=top_k)
        else:
            selected = []
    else:
        path = Path(vector_store_path).expanduser()
        if not vector_store_path.strip() or not (path / "index.faiss").is_file() or not (path / "metadata.json").is_file():
            raise ValueError("Choose a vector store directory containing index.faiss and metadata.json.")
        timings["load_seconds"] = perf_counter() - started
        yield "retrieve", None
        selected = create_retriever(str(path), embedding_model).retrieve(claim, k=top_k)
        available = None
    timings["retrieve_seconds"] = perf_counter() - started - timings["load_seconds"]
    yield "generate", None
    generation_start = perf_counter()
    result = verify_claim(claim, selected, llm=llm if llm is not None else get_llm(model_name=model_name), max_papers=top_k)
    if result.analyses and all(a.status == "failed" for a in result.analyses):
        raise RuntimeError("All paper analyses failed. Check that Ollama is running and the selected model is installed.")
    timings["generate_seconds"] = perf_counter() - generation_start
    timings["total_seconds"] = perf_counter() - started
    labels = {"supports": ("Supported", "supported"), "contradicts": ("Refuted", "refuted"),
              "mixed": ("Mixed evidence", "mixed"), "neutral": ("Neutral evidence", "caution"),
              "insufficient": ("Not enough evidence", "caution")}
    label, tone = labels[result.verdict]
    evidence = {e.paper_id: e for e in result.evidence}
    scores = {(c.paper_id, c.chunk_id): c.score for c in selected}
    sources = []
    for source in result.sources:
        entry = evidence[source.paper_id]
        sources.append({"rank": source.citation_id, "title": source.title,
            "year": source.year or "Unknown", "relevance": scores.get((source.paper_id, source.chunk_id)),
            "why_selected": entry.summary, "url": source.url or "", "stance": entry.stance,
            "study_type": entry.study_design, "page": source.page_number or "Unknown",
            "section": source.metadata.get("evidence_type", "Evidence excerpt"), "snippet": source.excerpt})
    yield "done", {
        "claim": claim, "verdict": {"label": label, "tone": tone},
        "confidence": {"label": result.confidence.title(), "rationale": result.confidence_reason},
        "evidence_summary": [f"{item.statement} [{', '.join(item.citation_ids)}]" for item in result.evidence_summary],
        "limitations": result.limitations, "answer": "",
        "sources": sources, "timings": timings,
        "metadata": {"model": model_name, "papers_available": available, "top_k": top_k,
                     "pipeline_mode": evidence_source, "retrieved_chunks": len(selected)},
        "assessment": result.model_dump(mode="json"),
    }


def run_pipeline(claim, top_k=4, model_name=DEFAULT_MODEL, **options):
    for stage, result in pipeline_events(claim, top_k, model_name, **options):
        if stage == "done":
            return result
