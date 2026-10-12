# Message to Baraa

Hi Baraa, I checked the diagram and updated the UI/output plan to match the proposed structure.

For the UI, I plan to show:

- Input: one rumor/claim text box.
- Loading/progress state: visible steps for scrape/load papers, RAG retrieval, LLM answer, and final response.
- Output:
  - verdict label: `Supported`, `Refuted`, `Mixed evidence`, or `Not enough evidence`
  - confidence label and rationale
  - evidence summary
  - evidence cards with stance, study type, relevance, page, section, snippet, and clickable source link
  - limitations
  - debug/timing metadata in collapsed sections

For the output contract, I suggest the backend returns:

```python
{
    "claim": "...",
    "verdict": {
        "label": "Not enough evidence",
        "tone": "caution",
        "rationale": "..."
    },
    "confidence": {
        "label": "Low",
        "rationale": "..."
    },
    "evidence_summary": ["...", "..."],
    "limitations": ["...", "..."],
    "answer": "...",
    "sources": [
        {
            "rank": 1,
            "title": "...",
            "authors": "...",
            "year": 2026,
            "url": "...",
            "doi": "...",
            "relevance": 0.91,
            "stance": "Supports | Contradicts | Mixed | Background",
            "study_type": "Meta-analysis | Cohort | Survey | Method paper",
            "page": 6,
            "section": "Results",
            "snippet": "...",
            "why_selected": "..."
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

My recommendation on the blue points:

- UI: loading/progress state and clickable references are already planned.
- Dataset: yes, we should store Semantic Scholar metadata plus chunk-level `page`, `section`, and `text`.
- RAG: start with max 4 evidence chunks. If no relevant evidence exists, return `Not enough evidence`.
- LLM: start with one model first for demo reliability. Add local/Gemini comparison or parallel independent requests only after the basic flow is stable.
- LangGraph: useful later, but I would not make it required for the first working integration.

I can plug the backend into the UI as long as it follows this output shape.
