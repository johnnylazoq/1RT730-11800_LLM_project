# 1RT730-11800 LLM Project

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Structure

```
src/llm_project/
    data/         # data loading & preprocessing
    models/       # model definitions
    training/      # training loops / scripts
    evaluation/    # evaluation & metrics
    utils/         # shared helpers
notebooks/         # exploratory notebooks
scripts/           # CLI entry points
tests/             # unit tests
configs/           # experiment configs
data/raw/          # raw datasets (gitignored)
data/processed/    # processed datasets (gitignored)
outputs/           # checkpoints & logs (gitignored)
```

## Claim verification pipeline

The implementation in `src/app/` accepts retrieved evidence; it does not perform
PDF extraction, embedding, vector storage, or vector retrieval. Existing scripts
in `scripts/` are preserved. The Semantic Scholar demo uses abstracts as test
chunks and also accepts the existing `top_5_papers.json` format.

Install just the downstream application and its tests:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
```

The workflow is `rank/group → bounded parallel paper analyses → LLM synthesis →
citation validation → FinalAssessment`. It reuses the existing Ollama client
and Pydantic models. Configure `OLLAMA_MODEL` (default
`frob/qwen3.5-instruct:9b`), `OLLAMA_BASE_URL` (default
`http://localhost:11434`), and `OLLAMA_TIMEOUT` (default 120 seconds per transport
operation). A running Ollama server with the selected model is required for live
analysis. Models must support structured JSON output. Threads bound concurrent
requests; actual inference parallelism depends on the Ollama server.

### Test input without RAG

Optionally collect one query using a **single** Semantic Scholar request (no
pagination or automatic retry). Set `SEMANTIC_SCHOLAR_API_KEY` if available:

```bash
claim-demo fetch --query 'coffee cardiovascular disease' --output data/raw/coffee.json
claim-demo verify --claim 'Coffee causes cardiovascular disease.' \
  --input data/raw/coffee.json --output outputs/coffee.json
```

An HTTP 429 means the service rate-limited the request. No results are fabricated.
Fetching refuses to overwrite an existing cache. Reuse saved data for subsequent
analysis runs. Records without abstracts are skipped; a PDF URL alone is not
usable evidence. This is an abstract adapter for testing, not a full RAG system.

A reproducible **synthetic**, non-scientific fixture is included:

```bash
claim-demo verify --claim 'Coffee causes cardiovascular disease.' \
  --input tests/fixtures/retrieved_chunks.json --format chunks \
  --output outputs/synthetic_smoke.json --workers 2
```

Automated tests use fake LLM responses and need neither network access nor Ollama.
Synthetic evidence is explicitly marked in the final limitations.

### Integration contract

```python
from app.graph.workflow import verify_claim

result = verify_claim(
    claim="The claim to evaluate",
    retrieved_chunks=[{
        "paper_id": "stable-paper-id",
        "chunk_id": "stable-chunk-id",
        "text": "Exact retrieved excerpt",
        "title": "Paper title",
        "authors": ["Author Name"],
        "doi": None,
        "url": None,
        "year": 2024,
        "page_number": 3,
        "score": 0.9,
        "metadata": {"section": "Results"},
    }],
    max_workers=3,
    retries=1,
    higher_score_better=True,
    max_papers=10,
    max_chunks_per_paper=8,
    max_chars_per_paper=24000,
    evidence_cutoff=None,
)
print(result.model_dump_json(indent=2))
```

Paper/chunk identifiers and nonblank text are required. Chunk IDs need only be
unique within each paper. Put additional upstream metadata under `metadata`.
Exact duplicate records are deduplicated; conflicting duplicate content or
paper metadata is rejected. Scores rank chunks and each paper's best chunk ranks
papers. Missing scores sort after scored records; ties preserve input order.
Use `higher_score_better=False` for distances. Ranking does not measure evidence
quality. Evidence budgets omit whole chunks, never truncate cited excerpts, and
report omissions in limitations. Cutoff is supplied by the caller, not inferred
from publication dates or the day the request was run.

Per-paper findings include stance, study design, quality considerations,
limitations, and chunk references. Failed calls are explicitly marked `failed`
and cannot provide citations; they are not scientific findings. Paper failures
allow partial synthesis, while an invalid aggregate fails with an exception
instead of returning unchecked citations. Empty/no-valid-evidence input yields
an `insufficient`, low-confidence response without an aggregation call.

The final JSON contains verdict, confidence and rationale, cited evidence
summary, evidence by paper, limitations, full analyses, and sources with exact
excerpts and available page/DOI/author metadata. Citation IDs are stable numbered
strings: render `citation_ids: ["1", "2"]` as `[1][2]` in a UI. Bibliographic
metadata is constructed from input records, not model output. Stances are
`supports`, `contradicts`, `mixed`, `neutral`, or `insufficient`; confidence is
`low`, `moderate`, or `high`. Confidence is a model judgment, not a calibrated
probability. Citation checks validate source identity and provenance, not whether
the source semantically proves every generated statement.

## RAG Pipeline (NEW)

The RAG pipeline enables ingesting documents and dynamically retrieving relevant evidence for claim verification.

### Components

- `src/app/rag/collect_papers.py` - Fetch papers from Semantic Scholar API
- `src/app/rag/extract_documents.py` - Parse PDFs to text
- `src/app/rag/chunk_documents.py` - Split documents into chunks
- `src/app/rag/embeddings.py` - Generate embeddings using sentence-transformers
- `src/app/rag/vector_store.py` - Store/retrieve vectors using FAISS
- `src/app/rag/ingest.py` - Orchestrate ingestion pipeline
- `src/app/rag/retriever.py` - Query vector store for relevant chunks
- `src/app/graph/nodes/rag.py` - LangGraph node for RAG retrieval

### Ingest Papers from Semantic Scholar API

```bash
claim-demo ingest-api \
  --query "vitamin C common cold" \
  --output vector_db_papers \
  --limit 20
```

### Ingest PDFs

```bash
claim-demo ingest-pdf \
  --input data/pdfs/ \
  --output vector_db_pdfs \
  --strategy paragraph
```

### Verify with RAG Retrieval

Automatically retrieve relevant evidence and verify:

```bash
claim-demo rag-verify \
  --claim "Vitamin C prevents the common cold" \
  --vector-store vector_db_papers \
  --output results.json \
  --top-k 10
```

### Programmatic RAG Usage

```python
from app.rag.retriever import create_retriever
from app.graph.workflow import verify_claim

retriever = create_retriever("vector_db_papers")
result = verify_claim(
    claim="Vitamin C prevents cold",
    retrieved_chunks=[],
    retriever=retriever,
    max_workers=3
)
print(result.model_dump_json(indent=2))
```

### Configuration

Environment variables:

- `SEMANTIC_SCHOLAR_API_KEY` - (Optional) API key for Semantic Scholar
- `OLLAMA_MODEL` - (Default: `frob/qwen3.5-instruct:9b`)
- `OLLAMA_BASE_URL` - (Default: `http://localhost:11434`)

### Dependencies

RAG pipeline requires:

```bash
pip install pypdf>=4 sentence-transformers>=3 faiss-cpu>=1.7
```
