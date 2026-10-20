# Rumor2Reality UI

The UI runs Semantic Scholar discovery → embeddings/FAISS retrieval → LangGraph
paper analysis → citation-validated synthesis with Ollama. No canned verdicts are used.

From the repository root (Python 3.11+):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[ui,test]"
ollama pull frob/qwen3.5-instruct:9b
python ui/rumor2reality/app.py
```

Ollama must be running. Open the local URL printed by Gradio. Enter a claim,
select the installed Ollama model, and run the check. The first retrieval downloads
the embedding model; Semantic Scholar may rate-limit requests. An optional
`SEMANTIC_SCHOLAR_API_KEY` improves API access.

Choose **Semantic Scholar** for fresh abstract discovery and a temporary index per
request. Choose **Existing vector store** and enter an absolute directory path to
use papers or PDFs already ingested with `claim-demo ingest-api` or
`claim-demo ingest-pdf`. The directory must contain `index.faiss` and `metadata.json`.
Use the same `EMBEDDING_MODEL` during ingestion and retrieval (default:
`all-MiniLM-L6-v2`). The retrieval control counts chunks, which may come from the
same paper. Similarity is a retrieval score, not a probability of truth.

`OLLAMA_MODEL`, `OLLAMA_BASE_URL`, and `OLLAMA_TIMEOUT` configure local inference.
The model textbox overrides `OLLAMA_MODEL` for that request. Each run displays
actual stage progress, validated citation IDs, evidence excerpts, limitations,
and timings. Empty evidence produces an insufficient-evidence assessment;
retrieval/service failures are shown as errors. No public sharing is enabled.

```powershell
python -m pytest tests/test_pipeline.py tests/test_ui_integration.py -q
```

Integration tests use controlled retrieval and a fake structured LLM with the real
LangGraph workflow. They do not certify live model quality or service availability.
