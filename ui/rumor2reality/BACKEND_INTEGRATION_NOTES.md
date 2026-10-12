# Backend Integration Notes

## Updates From Johnny

Johnny's branch appears to implement a paper-processing/correlation backend with:

- LangGraph workflow orchestration
- JSON paper loading
- chunking
- sentence-transformer embeddings
- FAISS vector search
- LLM-based paper-pair analysis
- JSON/CSV output

This is useful backend infrastructure, but the current branch is shaped as a
**paper correlation pipeline**, not yet a direct **rumor verification pipeline**.

## What The UI Needs

The Rumor2Reality UI expects a claim-level response:

```python
{
    "claim": "...",
    "verdict": {
        "label": "Supported | Refuted | Mixed evidence | Not enough evidence",
        "tone": "supported | refuted | mixed | caution",
        "rationale": "..."
    },
    "confidence": {
        "label": "Low | Moderate | High",
        "rationale": "..."
    },
    "evidence_summary": ["...", "..."],
    "limitations": ["...", "..."],
    "answer": "...",
    "sources": [
        {
            "rank": 1,
            "title": "...",
            "authors": ["..."],
            "year": 2026,
            "url": "...",
            "relevance": 0.91,
            "why_selected": "...",
            "stance": "Supports | Contradicts | Mixed | Background",
            "study_type": "Meta-analysis | Cohort | Survey | Method paper",
            "page": 6,
            "section": "Results",
            "snippet": "..."
        }
    ],
    "timings": {
        "scrape_or_load_papers": 0.2,
        "rag_retrieval": 0.2,
        "llm_generation": 0.2,
        "total": 0.6
    }
}
```

Salah's proposed backend response can be adapted with
`backend_adapter.normalize_backend_response()`.

## Suggested Backend Adapter Task

Johnny's pipeline should expose one function for the UI:

```python
def answer_claim(claim: str, top_k: int = 4) -> dict:
    ...
```

It should:

1. Retrieve relevant paper chunks for the claim.
2. Classify each chunk as `Supports`, `Contradicts`, `Mixed`, or `Background`.
3. Generate a claim-level verdict.
4. Return sources with page/section/snippet metadata.
5. Return timings.

## PR Review Notes For Johnny

- The README still describes several components as TODO/stubbed. Update it to match the actual implementation status.
- The branch is currently framed around correlations between papers. For Rumor2Reality, we need claim-to-evidence verification.
- The UI does not need autonomous agents. LangGraph as a workflow engine is fine.
- The graph should have a real runnable smoke test, not a skipped graph-build test.
- Check async/sync behavior in `build_graph.py`: some graph nodes are `async`, but wrappers appear to call them without `await`.
- Confirm whether `StateGraph` uses `START`/`END` constants from LangGraph rather than string literals.
- Add a sample output JSON for one rumor claim so UI integration can be tested without API keys.
- Add page/section metadata to chunks if possible, since the UI now displays clickable evidence references.

## Recommendation

Do not merge into `main` until:

- `python scripts/build_index.py` runs on sample JSON papers.
- `python scripts/run_graph.py --query "online rumors are created by bots"` produces JSON output.
- The output can be adapted into the UI contract.
- The README accurately explains how to run the branch.
