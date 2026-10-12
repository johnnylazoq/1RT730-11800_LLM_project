# Rumor2Reality

Rumor2Reality is a Gradio demo for an evidence-grounded rumor checking pipeline.
This first version gives the group a deployable UI and a stable backend contract.
The current backend is mocked so the scraping, RAG, and LLM stages can be replaced
incrementally.

## Project Structure

```text
rumor2reality/
  app.py            # Gradio UI and demo flow
  pipeline.py       # Backend contract and mock pipeline
  requirements.txt  # Dependencies for local run and Hugging Face Spaces
```

## Run Locally

```bash
python -m pip install -r requirements.txt
python app.py
```

Open the local URL printed by Gradio, usually `http://127.0.0.1:7860`.

## Backend Contract

The UI calls this function:

```python
run_pipeline(claim: str, top_k: int = 5, model_name: str = "mock-llm") -> dict
```

It should return:

```python
{
    "claim": "...",
    "verdict": {
        "label": "Not enough evidence",
        "tone": "caution",
        "rationale": "Retrieved papers are relevant background, but they do not directly prove or disprove the specific rumor."
    },
    "confidence": {
        "label": "Low",
        "rationale": "Sources are relevant background, but not direct evidence for the specific claim."
    },
    "evidence_summary": [
        "Retrieved background evidence from 4 paper(s).",
        "The current starter corpus explains methods and datasets, but does not yet contain claim-specific study findings."
    ],
    "limitations": [
        "Starter corpus contains paper-level metadata and summaries, not full PDF chunks yet.",
        "Current retrieval is keyword-based; embeddings or a vector index should replace it."
    ],
    "answer": "...",
    "sources": [
        {
            "rank": 1,
            "title": "...",
            "authors": "...",
            "year": 2026,
            "url": "...",
            "relevance": 0.91,
            "why_selected": "...",
            "abstract": "...",
            "stance": "Background",
            "study_type": "Survey",
            "page": 1,
            "section": "Overview",
            "snippet": "..."
        }
    ],
    "timings": {
        "scrape_or_load_papers": 0.2,
        "rag_retrieval": 0.2,
        "llm_generation": 0.2,
        "total": 0.6
    },
    "metadata": {
        "model": "mock-llm",
        "papers_available": 5,
        "top_k": 5,
        "pipeline_mode": "mock"
    }
}
```

Keep these keys stable when replacing the mock pipeline. That lets the UI and
deployment work stay separate from the model work.

Recommended final verdict labels:

- `Supported`
- `Refuted`
- `Mixed evidence`
- `Not enough evidence`

## Suggested Integration Plan

1. Replace `retrieve_papers()` with the scraper or a loader for pre-collected papers.
2. Add embeddings or keyword search for the RAG stage.
3. Replace `generate_answer()` with the chosen LLM call.
4. Keep timing values so the presentation can discuss latency and optimization.
5. Keep source rows visible so challengers can see what evidence the answer used.

If the backend returns the compact shape proposed by the group:

```python
{
    "final_answer": "...",
    "verdict": "...",
    "sources": [
        {
            "title": "...",
            "year": "...",
            "url": "...",
            "relevance_score": 0.91,
            "classification": "Supports",
            "explanation": "...",
            "reason_selected": "..."
        }
    ],
    "timing": {"llm_generation": 1.2}
}
```

use `backend_adapter.normalize_backend_response()` to convert it into the UI
contract without changing `app.py`.

## Deploy on Hugging Face Spaces

1. Create a new Hugging Face Space.
2. Select **Gradio** as the SDK.
3. Upload or push these files at the Space root:
   - `app.py`
   - `pipeline.py`
   - `requirements.txt`
4. Wait for the Space build to finish.

For API-based models, add keys in the Space settings as secrets. Do not commit
API keys to GitHub.

## Presentation Notes

Useful points for the UI/deploy contribution statement:

- Built a Gradio interface around the full system flow.
- Added visible pipeline status for scraping, retrieval, generation, and final answer.
- Added source table and debug contract to make the RAG behavior explainable.
- Prepared the project for Hugging Face Spaces deployment.
