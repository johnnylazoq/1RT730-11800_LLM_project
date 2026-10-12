from __future__ import annotations

from typing import Any


VERDICT_TONES = {
    "supported": "supported",
    "refuted": "refuted",
    "mixed evidence": "mixed",
    "mixed": "mixed",
    "not enough evidence": "caution",
    "evidence does not currently support claim": "caution",
}


def _verdict_from_backend(raw_verdict: Any) -> dict[str, str]:
    if isinstance(raw_verdict, dict):
        label = str(raw_verdict.get("label", "Not enough evidence"))
        rationale = str(raw_verdict.get("rationale", ""))
        tone = str(raw_verdict.get("tone", VERDICT_TONES.get(label.lower(), "caution")))
        return {"label": label, "tone": tone, "rationale": rationale}

    label = str(raw_verdict or "Not enough evidence")
    return {
        "label": label,
        "tone": VERDICT_TONES.get(label.lower(), "caution"),
        "rationale": "Verdict returned by the backend.",
    }


def normalize_backend_response(
    raw: dict[str, Any],
    claim: str,
    top_k: int,
    model_name: str,
) -> dict[str, Any]:
    """Normalize teammate backend output into the UI contract.

    This accepts the compact backend shape proposed by Salah:
    final_answer, verdict, sources[], and timing.
    """
    sources = []
    for index, source in enumerate(raw.get("sources", []), start=1):
        relevance = source.get("relevance_score", source.get("relevance", 0.0))
        try:
            relevance = float(relevance)
        except (TypeError, ValueError):
            relevance = 0.0

        classification = str(source.get("classification", source.get("stance", "Background")))
        explanation = str(source.get("explanation", source.get("reason_selected", "")))
        reason_selected = str(source.get("reason_selected", explanation or "Selected by backend retrieval."))

        sources.append(
            {
                "rank": source.get("rank", index),
                "title": str(source.get("title", "Untitled source")),
                "authors": source.get("authors", []),
                "year": source.get("year", "unknown"),
                "url": str(source.get("url", "")),
                "relevance": max(0.0, min(1.0, relevance)),
                "why_selected": reason_selected,
                "abstract": str(source.get("abstract", "")),
                "stance": classification,
                "study_type": str(source.get("study_type", source.get("paper_type", "Paper"))),
                "page": source.get("page", "?"),
                "section": str(source.get("section", source.get("section_name", "Evidence"))),
                "snippet": explanation,
            }
        )

    verdict = _verdict_from_backend(raw.get("verdict"))
    answer = str(raw.get("final_answer", raw.get("answer", ""))).strip()
    if not answer:
        answer = "The backend did not return a final answer."

    confidence = raw.get("confidence")
    if isinstance(confidence, dict):
        normalized_confidence = {
            "label": str(confidence.get("label", "Unknown")),
            "rationale": str(confidence.get("rationale", "")),
        }
    else:
        normalized_confidence = {
            "label": str(confidence or "Unknown"),
            "rationale": "Confidence returned by the backend.",
        }

    timing = raw.get("timing", raw.get("timings", {}))
    if "llm_generation" in timing and "llm_generation" not in raw.get("timings", {}):
        timing = dict(timing)

    evidence_summary = raw.get("evidence_summary")
    if not evidence_summary:
        evidence_summary = [
            f"Backend returned {len(sources)} source(s) for this claim.",
            "Each source includes classification, relevance, explanation, and reason selected when available.",
        ]

    limitations = raw.get("limitations")
    if not limitations:
        limitations = [
            "Limitations were not returned by the backend.",
            "UI can display limitations once the LLM or RAG pipeline provides them.",
        ]

    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": normalized_confidence,
        "evidence_summary": evidence_summary,
        "limitations": limitations,
        "answer": answer,
        "sources": sources[:top_k],
        "timings": timing,
        "metadata": {
            "model": model_name,
            "papers_available": len(sources),
            "top_k": top_k,
            "pipeline_mode": "external-backend",
        },
    }
