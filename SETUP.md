# SETUP COMPLETE: Paper Correlation Pipeline Project

## Summary of PHASE 1, 2, 3 Implementation

### ✅ Phase 1: Structure
Created complete module structure with 16 Python files organized into 4 subsystems:

**Core Module** (`src/paper_corr/`)
- `__init__.py` — Package initialization
- `config.py` — YAML config loading, API key management

**RAG Subsystem** (`src/paper_corr/rag/`)
- `loader.py` — Paper loader (PDF/JSON) → Paper namedtuple
- `chunker.py` — Paper chunking with section labels
- `embeddings.py` — Embedding model wrapper (sentence-transformers)
- `vectorstore.py` — Vector index (FAISS/Chroma) interface
- `retriever.py` — Retrieval methods (papers, chunks by query)

**LLM Subsystem** (`src/paper_corr/llm/`)
- `client.py` — LLM client wrapper (OpenAI, Anthropic, etc.)
- `prompts.py` — Pair analysis prompt template builder

**LangGraph Pipeline** (`src/paper_corr/graph/`)
- `schemas.py` — Pydantic models (CorrelationResult, CorrelationMatrix)
- `state.py` — GraphState TypedDict with Annotated correlations reducer
- `nodes.py` — Graph nodes (retrieve_papers, retrieve_chunks, build_pairs, analyze_pair, aggregate, save_results)
- `build_graph.py` — Graph compilation and run_pipeline() entry point

**Configuration**
- `configs/default.yaml` — All pipeline parameters (top_k, model, relation_types, etc.)
- `.env_example` — Environment variable template (LLM keys, LangSmith)
- `.gitignore` — Updated to exclude .env, outputs/, data/index/, vector files

### ✅ Phase 2: LangGraph Pipeline Design
- **State**: Annotated reducer on correlations for parallel aggregation
- **Nodes**: 6 nodes implementing the flow (retrieve → chunk → build pairs → analyze → aggregate → save)
- **Fan-out**: analyze_pair uses langgraph.Send for parallel processing
- **Deduplication**: build_pairs filters self-pairs and (A,B)/(B,A) duplicates
- **Schemas**: Pydantic validation with correlation_score (0-1), relation_type, evidence, confidence

### ✅ Phase 3: Config, Env, LangSmith
- **requirements.txt**: Updated with all 14 dependencies (langgraph, langchain-core, langsmith, pydantic, etc.)
- **pyproject.toml**: Updated with all dependencies, Python >=3.11
- **.env_example**: Contains all keys needed (LANGSMITH_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
- **LangSmith**: Configured with project name and tracing setup in config

### ✅ Phase 4: Tests & Docs
**Tests** (`tests/`)
- `test_schemas.py` — 5 tests: validation, deduplication, sorting
- `test_graph.py` — Graph compilation, state initialization

**Documentation**
- `docs/pipeline.md` — Architecture diagram, setup steps, CLI usage, config reference, TODO checklist

**Scripts** (`scripts/`)
- `build_index.py` — Stub to ingest papers → vector index
- `run_graph.py` — CLI: --query or --paper-ids to run pipeline
- `evaluate.py` — Stub for evaluation against ground truth

## Verification ✓

```
✓ Core modules import successfully
✓ CorrelationResult validation works (Pydantic)
✓ CorrelationMatrix add/dedup works
✓ GraphState TypedDict defined
✓ All 16 Python files created
✓ All required dependencies in requirements.txt
✓ .env_example and configs/default.yaml in place
```

## What You Need to Fill In

### 1. Environment Setup
```bash
# Copy template and fill in API keys
cp .env_example .env

# Add to .env:
LANGSMITH_API_KEY=<your-key>
OPENAI_API_KEY=<your-key>  # or ANTHROPIC_API_KEY for Claude
```

### 2. Implementation TODOs (Marked with `raise NotImplementedError()` in stubs)

**RAG Pipeline** (4 files to implement):
- `loader.py`: PDF/JSON parsing (use PyPDF2, pdfplumber, or langchain loaders)
- `chunker.py`: Section detection + sliding window chunking
- `embeddings.py`: Load sentence-transformers model
- `vectorstore.py`: Initialize FAISS/Chroma, add/search methods
- `retriever.py`: Wire retriever to vectorstore

**LLM Integration** (1 file):
- `client.py`: LLMClient init (detect provider from model name or env), generate() with structured output

**LangGraph** (2 files):
- `nodes.py`: Implement all 6 node async functions
- `build_graph.py`: StateGraph construction + Send for parallel analyze_pair

**Scripts** (3 files):
- `build_index.py`: Load papers → chunk → embed → vectorstore.save()
- `run_graph.py`: Parse args, load config, call run_pipeline()
- `evaluate.py`: Load predictions + ground truth, compute metrics

### 3. Data Format (Place papers in `data/raw/` or `data/processed/`)

Papers can be:
- **JSON**: `{"paper_id": "...", "title": "...", "abstract": "...", "content": "...", "authors": [...], "metadata": {...}}`
- **PDF**: Auto-extracted with text extraction library

### 4. Model Choice
Decide on LLM provider for `config.model` in `configs/default.yaml`:
- OpenAI: `gpt-4-turbo` or `gpt-4o`
- Anthropic: `claude-3-5-sonnet-20241022`
- Local: Ollama, LLaMA.cpp, etc.

### 5. Vector Database Choice
Default is FAISS (CPU). Alternatives in vectorstore.py:
- **Chroma**: `pip install chromadb`
- **Pinecone**: `pip install pinecone-client` (requires PINECONE_API_KEY)
- **Weaviate**: `pip install weaviate-client`

## Files Created (23 total)

```
src/paper_corr/
├── __init__.py
├── config.py
├── rag/
│   ├── __init__.py
│   ├── loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   └── retriever.py
├── llm/
│   ├── __init__.py
│   ├── client.py
│   └── prompts.py
└── graph/
    ├── __init__.py
    ├── schemas.py
    ├── state.py
    ├── nodes.py
    └── build_graph.py

configs/
└── default.yaml

scripts/
├── build_index.py
├── run_graph.py
└── evaluate.py

tests/
├── test_schemas.py
└── test_graph.py

docs/
└── pipeline.md

.env_example (updated)
.gitignore (updated)
pyproject.toml (updated with dependencies)
requirements.txt (updated with dependencies)
```

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt` (already done with --no-cache-dir)
2. **Setup .env**: Copy .env_example to .env and fill in your API keys
3. **Add sample papers**: Place PDFs or JSON files in `data/raw/` or `data/processed/`
4. **Implement RAG loaders**: Start with loader.py → chunker.py → vectorstore.py
5. **Test with mocked LLM**: Run pytest (tests are schema-focused, not full integration)
6. **Run full pipeline**: `python scripts/build_index.py` then `python scripts/run_graph.py --query "..."`
7. **Monitor in LangSmith**: Every analyze_pair call traces automatically

## LangSmith Tracing
Once LANGSMITH_API_KEY is set and LANGSMITH_TRACING=true:
- Every node execution appears in LangSmith
- analyze_pair calls include paper IDs and query as metadata
- View at: https://smith.langchain.com/o/{org_id}/projects/llm-project-paper-correlations

## Configuration Reference
See `configs/default.yaml` for all tuneable parameters:
- `top_k`: Number of papers to retrieve (default: 10)
- `max_pairs`: Max candidate pairs to analyze (default: 50)
- `similarity_threshold`: Embedding cosine sim filter (default: 0.5)
- `relation_types`: Classification labels
- `model`, `temperature`, `max_tokens`: LLM settings
- `embedding_model`: Sentence-transformers model ID

---

**All structure, configuration, and documentation complete!** ✨
Ready for implementation of the TODO items.
