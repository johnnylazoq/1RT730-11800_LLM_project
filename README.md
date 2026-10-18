# 1RT730-11800 LLM Project: Rumor-to-Evidence Claim Verification

**Automated scientific claim verification using RAG and LLMs**

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Setup & Installation](#setup--installation)
4. [Quick Start](#quick-start)
5. [Core Components](#core-components)
6. [Claim Verification Pipeline](#claim-verification-pipeline)
7. [RAG Pipeline (NEW)](#rag-pipeline-new)
8. [CLI Commands](#cli-commands)
9. [Programmatic API](#programmatic-api)
10. [Testing](#testing)
11. [Configuration](#configuration)
12. [Project Structure](#project-structure)

---

## Overview

This project implements an end-to-end system for verifying scientific claims against research evidence:

1. **OFFLINE**: Ingest documents (PDFs, APIs) → Extract → Chunk → Embed → Vector Store
2. **ONLINE**: Query claims → Retrieve evidence → Parallel LLM analysis → Synthesis → Verdict

**Key Features:**
- ✅ Semantic Scholar API integration for paper discovery
- ✅ PDF extraction and intelligent chunking
- ✅ Dense embeddings (sentence-transformers, 384D)
- ✅ FAISS vector store for fast retrieval
- ✅ Parallel paper analysis with Qwen/Ollama LLM
- ✅ Evidence synthesis with confidence scores
- ✅ Citation validation
- ✅ LangGraph workflow orchestration

---

## Architecture

### System Diagram

```
╔════════════════════════════════════════════════════════════════╗
║                    RUMOR-TO-EVIDENCE SYSTEM                   ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  OFFLINE STAGE (Document Ingestion)                           ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                          ║
║  API/PDFs                                                      ║
║    ↓                                                            ║
║  ┌─ collect_papers.py (Semantic Scholar API)                  ║
║  ├─ extract_documents.py (PDF → text, PyPDF)                  ║
║  ├─ chunk_documents.py (Paragraph & fixed-size)               ║
║  ├─ embeddings.py (sentence-transformers, 384D)               ║
║  ├─ vector_store.py (FAISS vector DB)                         ║
║  └─ ingest.py (Pipeline orchestration)                        ║
║    ↓                                                            ║
║  Vector Store (on disk)                                        ║
║    index.faiss + metadata.json                                 ║
║                                                                ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    ║
║                                                                ║
║  ONLINE STAGE (Claim Verification)                            ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                            ║
║  Claim Input                                                   ║
║    ↓                                                            ║
║  retrieve_rag.py (NEW - Query vector store)                   ║
║    ↓                                                            ║
║  LangGraph Workflow:                                           ║
║    ├─ group_papers (dedup, rank)                              ║
║    ├─ analyze_papers (parallel LLM)                           ║
║    ├─ aggregate_evidence (synthesis)                          ║
║    └─ citations (validation)                                  ║
║    ↓                                                            ║
║  FinalAssessment                                               ║
║    {verdict, confidence, sources, citations, limitations}     ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

### Component Stack

| Layer | Component | Technology |
|-------|-----------|-----------|
| **Vector DB** | FAISS | IndexFlatL2 (L2 distance) |
| **Embeddings** | sentence-transformers | all-MiniLM-L6-v2 (384D) |
| **Chunking** | Custom | Paragraph-aware or fixed-size |
| **Extraction** | PyPDF | PDF → text parsing |
| **Collection** | Semantic Scholar API | Academic paper discovery |
| **LLM** | Ollama | Qwen3.5-instruct (configurable) |
| **Orchestration** | LangGraph | State-based workflow |
| **CLI** | Argparse | Command-line interface |

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- Ollama with Qwen model (or compatible LLM endpoint)
- ~2GB disk space for vector store

### Installation

1. **Clone and navigate to project:**
   ```bash
   cd /path/to/1RT730-11800_LLM_project
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e '.[test]'
   ```

4. **Verify installation:**
   ```bash
   claim-demo --help
   ```

### Optional: API Key

Set Semantic Scholar API key (optional but recommended):
```bash
export SEMANTIC_SCHOLAR_API_KEY="your-api-key-here"
```

---

## Quick Start

### Scenario: Verify "Vitamin C prevents the common cold"

#### Step 1: Ingest Papers from Semantic Scholar

```bash
claim-demo ingest-api \
  --query "vitamin C prevents common cold" \
  --output vector_db_vitaminc \
  --limit 20
```

**Output:**
- `vector_db_vitaminc/index.faiss` - Vector index
- `vector_db_vitaminc/metadata.json` - Document metadata

#### Step 2: Verify Claim Using RAG

```bash
claim-demo rag-verify \
  --claim "Vitamin C prevents the common cold" \
  --vector-store vector_db_vitaminc \
  --output results.json \
  --top-k 10 \
  --workers 3
```

#### Step 3: View Results

```bash
cat results.json | jq '.result.assessment'
```

**Output:**
```json
{
  "verdict": "Mixed evidence",
  "confidence": "moderate",
  "rationale": "While some studies support vitamin C's role...",
  "evidence_summary": [...],
  "limitations": [...]
}
```

---

## Core Components

### 1. RAG Pipeline (NEW)

#### 1.1 collect_papers.py
**Purpose:** Fetch papers from Semantic Scholar API

```python
from app.rag.collect_papers import fetch_papers, papers_to_chunks

papers = fetch_papers("vitamin C cold", limit=20)
chunks = papers_to_chunks(papers)  # → RetrievedChunk objects
```

#### 1.2 extract_documents.py
**Purpose:** Extract text from PDFs

```python
from app.rag.extract_documents import extract_text_from_pdf, extract_from_multiple_pdfs

text = extract_text_from_pdf("paper.pdf")
texts = extract_from_multiple_pdfs("pdfs/")
```

#### 1.3 chunk_documents.py
**Purpose:** Split text into chunks for embedding

```python
from app.rag.chunk_documents import chunk_text, chunk_by_paragraphs

# Fixed-size chunking with overlap
chunks = chunk_text(text, source="doc1", chunk_size=1000, overlap=100)

# Paragraph-aware chunking
chunks = chunk_by_paragraphs(text, source="doc1", max_chunk_size=2000)
```

#### 1.4 embeddings.py
**Purpose:** Generate dense embeddings

```python
from app.rag.embeddings import EmbeddingModel

model = EmbeddingModel("all-MiniLM-L6-v2")  # 384D embeddings
embeddings = model.embed(["text1", "text2"])  # (2, 384) array
```

#### 1.5 vector_store.py
**Purpose:** Store and retrieve vectors (FAISS)

```python
from app.rag.vector_store import VectorStore

store = VectorStore(embedding_dim=384)
store.add(embeddings, metadata)
results = store.search(query_embedding, k=5)  # [(id, distance, metadata), ...]
store.save("vector_db")
```

#### 1.6 ingest.py
**Purpose:** Orchestrate entire ingestion pipeline

```python
from app.rag.ingest import ingest_papers_from_api, ingest_pdfs

# From API
vector_store = ingest_papers_from_api("query", vector_store_path="vector_db", limit=20)

# From PDFs
vector_store = ingest_pdfs("pdfs/", vector_store_path="vector_db", chunk_strategy="paragraph")
```

#### 1.7 retriever.py
**Purpose:** Query vector store and return relevant chunks

```python
from app.rag.retriever import create_retriever

retriever = create_retriever("vector_db")
chunks = retriever.retrieve("Vitamin C prevents cold", k=10)  # → RetrievedChunk[]
```

#### 1.8 rag.py (LangGraph Node)
**Purpose:** RAG node in workflow graph

```python
from app.graph.nodes.rag import retrieve_evidence

# Automatically called when retriever is provided to workflow
```

### 2. Claim Verification Pipeline (Existing)

#### 2.1 group_papers.py
- Deduplicates chunks by paper
- Ranks papers by relevance score
- Limits papers by evidence budget
- Groups chunks per paper

#### 2.2 analyze_paper.py
- Calls LLM for each paper
- Extracts stance (supports/contradicts/mixed/neutral/insufficient)
- Generates explanation
- Handles retries with feedback

#### 2.3 aggregate_evidence.py
- Synthesizes findings from all papers
- Generates final verdict
- Computes confidence level (low/moderate/high)
- Formats evidence summary

#### 2.4 citations.py
- Validates citations match source text
- Prevents hallucinations
- Maps citation IDs to sources

#### 2.5 llm/
- **client.py**: Ollama/Qwen HTTP interface
- **prompts.py**: LLM prompt templates
- **schemas.py**: Pydantic data models
- **citations.py**: Citation validation logic

---

## Claim Verification Pipeline

### Data Flow

```
User Input (Claim)
    ↓
retrieve_rag (Query vector store if provided)
    ↓
group_papers (Dedup, rank, budget)
    ↓
analyze_papers (Parallel LLM analysis × N papers)
    ├─ Paper 1: stance, evidence, explanation
    ├─ Paper 2: stance, evidence, explanation
    └─ Paper N: stance, evidence, explanation
    ↓
aggregate_evidence (LLM synthesis)
    ↓
FinalAssessment
    ├─ verdict: Supported/Mixed/Refuted/Insufficient
    ├─ confidence: low/moderate/high
    ├─ rationale: Summary of findings
    ├─ evidence_summary: Key findings per paper
    ├─ limitations: Data/methodology notes
    ├─ sources: Full metadata + excerpts
    └─ citations: Validated citation IDs → sources
```

### Example Output

```json
{
  "verdict": "Mixed evidence",
  "confidence": "moderate",
  "rationale": "While some RCTs support...",
  "evidence_summary": [
    {
      "stance": "supports",
      "paper_id": "12345",
      "title": "Vitamin C and Common Cold...",
      "explanation": "Found 50% reduction in...",
      "cited_excerpts": [
        {
          "citation_id": "1",
          "text": "Our meta-analysis shows...",
          "page": 3
        }
      ]
    }
  ],
  "limitations": [
    "Evidence quality varies",
    "Limited to English publications"
  ],
  "sources": [...]
}
```

---

## RAG Pipeline (NEW)

### Offline Ingestion

**Three ingestion methods:**

#### Method 1: From Semantic Scholar API
```bash
claim-demo ingest-api \
  --query "your research topic" \
  --output vector_db \
  --limit 50
```

#### Method 2: From Local PDFs
```bash
claim-demo ingest-pdf \
  --input /path/to/pdfs/ \
  --output vector_db \
  --strategy paragraph
```

#### Method 3: Programmatically
```python
from app.rag.ingest import ingest_papers_from_api, ingest_pdfs

# From API
ingest_papers_from_api("vitamin C", limit=20, vector_store_path="vector_db")

# From PDFs
ingest_pdfs("pdfs/", vector_store_path="vector_db", chunk_strategy="paragraph")
```

### Online Retrieval

**Automatic (via CLI):**
```bash
claim-demo rag-verify --claim "..." --vector-store vector_db --output results.json
```

**Programmatic:**
```python
from app.rag.retriever import create_retriever
from app.graph.workflow import verify_claim

retriever = create_retriever("vector_db")
result = verify_claim(
    claim="Vitamin C prevents cold",
    retrieved_chunks=[],  # Can be empty
    retriever=retriever,
    max_workers=3
)
```

### Vector Store Management

**Save and load:**
```python
from app.rag.vector_store import VectorStore

store = VectorStore()
store.add(embeddings, metadata)
store.save("vector_db")

store.load("vector_db")
results = store.search(query_embedding, k=5)
```

---

## CLI Commands

### Ingestion Commands

#### ingest-api
Fetch papers from Semantic Scholar API and build vector store.

```bash
claim-demo ingest-api \
  --query "QUERY_STRING" \
  --output OUTPUT_DIR \
  --limit NUM_PAPERS
```

**Arguments:**
- `--query`: Search query (required)
- `--output`: Vector store directory (default: `vector_db`)
- `--limit`: Max papers to fetch (default: 10)

#### ingest-pdf
Extract PDFs and build vector store.

```bash
claim-demo ingest-pdf \
  --input PDF_DIRECTORY \
  --output OUTPUT_DIR \
  --strategy {paragraph,fixed}
```

**Arguments:**
- `--input`: Directory with PDFs (required)
- `--output`: Vector store directory (default: `vector_db`)
- `--strategy`: Chunking strategy (default: `paragraph`)
  - `paragraph`: Preserves semantic boundaries
  - `fixed`: Fixed-size chunks with overlap

### Verification Commands

#### rag-verify
Verify claim using RAG retrieval.

```bash
claim-demo rag-verify \
  --claim "YOUR_CLAIM" \
  --vector-store VECTOR_DB_DIR \
  --output OUTPUT_JSON \
  --top-k NUM_CHUNKS \
  --workers NUM_WORKERS
```

**Arguments:**
- `--claim`: Claim to verify (required)
- `--vector-store`: Path to vector store (default: `vector_db`)
- `--output`: Output JSON file (required)
- `--top-k`: Number of chunks to retrieve (default: 10)
- `--workers`: Parallel workers for LLM (default: 3)

#### verify (Legacy)
Verify claim from JSON file (backward compatible).

```bash
claim-demo verify \
  --claim "YOUR_CLAIM" \
  --input INPUT_JSON \
  --output OUTPUT_JSON \
  --format {abstracts,chunks} \
  --workers NUM_WORKERS
```

#### fetch (Legacy)
Fetch papers from Semantic Scholar (one-time, no pagination).

```bash
claim-demo fetch \
  --query "QUERY_STRING" \
  --output OUTPUT_JSON
```

---

## Programmatic API

### Example 1: Complete End-to-End

```python
from app.rag.ingest import ingest_papers_from_api
from app.rag.retriever import create_retriever
from app.graph.workflow import verify_claim

# Step 1: Ingest papers (one time)
ingest_papers_from_api("vitamin C cold", limit=20, vector_store_path="vc_db")

# Step 2: Create retriever
retriever = create_retriever("vc_db")

# Step 3: Verify claim
result = verify_claim(
    claim="Vitamin C prevents the common cold",
    retrieved_chunks=[],
    retriever=retriever,
    max_workers=3
)

# Step 4: Access results
print(f"Verdict: {result.assessment.verdict}")
print(f"Confidence: {result.assessment.confidence}")
print(f"Rationale: {result.assessment.rationale}")
```

### Example 2: Custom Chunking Strategy

```python
from app.rag.extract_documents import extract_text_from_pdf
from app.rag.chunk_documents import chunk_by_paragraphs
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore
from app.rag.ingest import ingest_chunks
from app.llm.schemas import RetrievedChunk

# Extract PDF
text = extract_text_from_pdf("paper.pdf")

# Custom chunking
chunks = chunk_by_paragraphs(text, source="paper.pdf", max_chunk_size=3000)

# Convert to RetrievedChunk
retrieved_chunks = [
    RetrievedChunk(
        paper_id="paper.pdf",
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        title="Research Paper",
        authors=["Author Name"],
        metadata={"evidence_type": "pdf_chunk"}
    )
    for chunk in chunks
]

# Ingest and verify
ingest_chunks(retrieved_chunks, vector_store_path="custom_db")
```

### Example 3: Hybrid Workflow (RAG + JSON)

```python
import json
from app.rag.retriever import create_retriever
from app.graph.workflow import verify_claim

# Use both vector store and static papers
retriever = create_retriever("vector_db")
static_chunks = json.load(open("top_papers.json"))

# Retriever has priority, static papers are fallback
result = verify_claim(
    claim="...",
    retrieved_chunks=static_chunks,
    retriever=retriever
)
```

---

## Testing

### Run Unit Tests

```bash
pytest tests/test_rag.py -v
```

### Test Coverage

- Text chunking (fixed and paragraph-based)
- Embedding generation
- Similarity computation
- Vector store operations (add, search, persistence)
- Workflow integration

### Example: Manual Testing

```bash
# Step 1: Build a small vector store
claim-demo ingest-api --query "coffee cardiovascular" --output test_db --limit 5

# Step 2: Verify with synthetic data
pytest tests/test_pipeline.py -v

# Step 3: Full end-to-end test
claim-demo rag-verify \
  --claim "Coffee causes heart disease" \
  --vector-store test_db \
  --output test_results.json \
  --top-k 5 \
  --workers 2
```

---

## Configuration

### Environment Variables

```bash
# Semantic Scholar API (optional but recommended)
export SEMANTIC_SCHOLAR_API_KEY="your-api-key"

# LLM Configuration
export OLLAMA_MODEL="frob/qwen3.5-instruct:9b"
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_TIMEOUT=120

# Optional: Embedding model
export EMBEDDING_MODEL="all-MiniLM-L6-v2"  # Default
```

### Ollama Setup

```bash
# Install Ollama: https://ollama.ai

# Pull Qwen model (first time)
ollama pull frob/qwen3.5-instruct:9b

# Start Ollama server
ollama serve

# In another terminal, test:
curl http://localhost:11434/api/generate \
  -d '{"model":"frob/qwen3.5-instruct:9b","prompt":"hello"}'
```

### pyproject.toml Settings

Core dependencies (installed automatically):
```toml
pydantic>=2,<3                 # Data validation
langgraph>=1,<2                # Workflow orchestration
langchain-core>=1,<2           # LLM framework
langchain-ollama>=1,<2         # Ollama integration
pypdf>=4,<5                    # PDF parsing
sentence-transformers>=3,<4    # Embeddings
faiss-cpu>=1.7,<2              # Vector DB
numpy>=1.24,<2                 # Numerical ops
```

Optional:
```bash
pip install faiss-gpu  # GPU acceleration instead of faiss-cpu
```

---

## Project Structure

```
src/app/
├── graph/
│   ├── workflow.py             Main LangGraph workflow
│   ├── state.py                GraphState TypedDict
│   └── nodes/
│       ├── rag.py              ✨ RAG retrieval node (NEW)
│       ├── group_papers.py      Dedup & rank
│       ├── analyze_paper.py     Single paper analysis
│       └── aggregate_evidence.py Synthesis
│
├── rag/                         ✨ RAG PIPELINE (NEW)
│   ├── collect_papers.py        API integration
│   ├── extract_documents.py     PDF parsing
│   ├── chunk_documents.py       Text chunking
│   ├── embeddings.py            Embedding generation
│   ├── vector_store.py          FAISS vector DB
│   ├── ingest.py                Pipeline orchestration
│   └── retriever.py             Query vector store
│
├── llm/
│   ├── client.py                Ollama interface
│   ├── prompts.py               LLM templates
│   ├── schemas.py               Pydantic models
│   └── citations.py             Citation validation
│
└── demo.py                      CLI entry point

tests/
├── test_pipeline.py             Integration tests
└── test_rag.py                  ✨ RAG unit tests (NEW)

data/
├── raw/                         Input data directory
└── processed/                   Processed datasets

outputs/
├── checkpoints/                 Model checkpoints
└── logs/                        Execution logs

vector_db/                       Vector store (generated)
├── index.faiss
└── metadata.json
```

---

## Workflow Summary

### Without RAG (JSON-Based)
```
User provides: JSON file with papers
         ↓
demo.py fetch/verify
         ↓
LangGraph: group → analyze → aggregate
         ↓
FinalAssessment
```

### With RAG (Vector-Based) ✨
```
User provides: Claim + Vector store path
         ↓
Automatic retrieval from vector store (NEW)
         ↓
LangGraph: retrieve_rag → group → analyze → aggregate
         ↓
FinalAssessment
```

---

## Key Differences: Old vs New

| Aspect | Old (JSON) | New (RAG) |
|--------|-----------|----------|
| **Input** | JSON file with papers | Claim + vector store |
| **Retrieval** | Manual / predetermined | Automatic / dynamic |
| **Scalability** | Limited to pre-selected papers | Unlimited document collection |
| **Speed** | Fast (small datasets) | Slower initial ingest, fast retrieval |
| **Flexibility** | Static papers | Any document source (PDF/API) |
| **Accuracy** | Depends on initial selection | Better relevance matching |
| **Cost** | Low (one-time) | Moderate (API calls + compute) |

---

## Architecture Advantages

✅ **Modular Design** - Each component is independent and testable  
✅ **Scalable** - FAISS supports millions of vectors  
✅ **Flexible** - Swap embedding models, chunking strategies, vector DBs  
✅ **Persistent** - Save/load vector stores to disk  
✅ **Type-Safe** - Full type hints throughout  
✅ **Error Handling** - Graceful degradation and clear messages  
✅ **Backward Compatible** - Old JSON workflow still works  
✅ **Production Ready** - No stubs, fully implemented  

---

## Future Enhancements

### Short Term (1-2 weeks)
- [ ] Hybrid search (vector similarity + BM25 keyword matching)
- [ ] Result caching for repeated queries
- [ ] Metadata filtering (year, author, publication)

### Medium Term (1 month)
- [ ] Alternative vector DBs (Pinecone, Milvus, Chroma)
- [ ] Advanced chunking (tree-based hierarchy, summaries)
- [ ] Multi-modal support (figures, tables)

### Long Term (2+ months)
- [ ] Interactive refinement UI
- [ ] Benchmark dataset creation
- [ ] Model fine-tuning on domain data
- [ ] Real-time fact-checking pipeline

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'faiss'"
```bash
pip install faiss-cpu
# Or for GPU:
pip install faiss-gpu
```

### "Connection refused" when calling Ollama
```bash
# Make sure Ollama server is running:
ollama serve

# Check port 11434:
curl http://localhost:11434/api/tags
```

### "No chunks found for query"
- Check vector store exists: `ls vector_db/`
- Verify query is similar to ingested documents
- Increase `--top-k` parameter
- Re-ingest documents if needed

### Out of memory during embedding
- Reduce batch size: Process fewer documents at once
- Use CPU-only embedding: `faiss-cpu` instead of `faiss-gpu`
- Reduce chunk size to generate fewer embeddings

---

## References

- **LangGraph:** https://github.com/langchain-ai/langgraph
- **Sentence-Transformers:** https://huggingface.co/sentence-transformers
- **FAISS:** https://faiss.ai/
- **Semantic Scholar API:** https://www.semanticscholar.org/product/api
- **Ollama:** https://ollama.ai/
- **Qwen Model:** https://huggingface.co/Qwen/Qwen3.5-32B-Instruct

---

## Contributing

To contribute to this project:

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes and test: `pytest tests/ -v`
3. Commit with descriptive messages
4. Push and create a pull request

---

## License

Part of course 1RT730 - Prepared by Johnny Lazo with Claude AI assistance.

---

## Contact

For questions or issues, open an issue on the project repository.

---

**Last Updated:** October 2026
**Status:** ✅ Production Ready
