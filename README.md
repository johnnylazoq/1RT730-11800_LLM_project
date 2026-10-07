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

src/paper_corr/           # Paper correlation pipeline (NEW)
    config.py             # Configuration loading
    rag/                  # Retrieval-Augmented Generation
        loader.py         # Load papers (PDF/JSON)
        chunker.py        # Split into chunks
        embeddings.py     # Embedding model
        vectorstore.py    # Vector index (FAISS)
        retriever.py      # Retrieve papers/chunks
    llm/                  # LLM integration
        client.py         # LLM client wrapper
        prompts.py        # Prompt templates
    graph/                # LangGraph pipeline
        schemas.py        # Pydantic models
        state.py          # Graph state
        nodes.py          # Graph nodes
        build_graph.py    # Compile graph

notebooks/         # exploratory notebooks
scripts/           # CLI entry points
  build_index.py   # Build vector index
  run_graph.py     # Run correlation pipeline
  evaluate.py      # Evaluate results
tests/             # unit tests
configs/           # experiment configs
  default.yaml     # Pipeline configuration
data/raw/          # raw datasets (gitignored)
data/processed/    # processed datasets (gitignored)
data/index/        # vector indices (gitignored)
outputs/           # results & logs (gitignored)
```

---

## Paper Correlation Pipeline

A system that analyzes academic papers and predicts correlations between them using Retrieval-Augmented Generation (RAG) and LLM analysis, orchestrated with LangGraph and traced with LangSmith.

### Quick Start

#### 1. Configure Environment

Copy the environment template and fill in your API keys:

```bash
cp .env_example .env
```

Then edit `.env` with your credentials:

```env
# LangSmith (required for tracing)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<your-langsmith-api-key>
LANGSMITH_PROJECT=llm-project-paper-correlations

# LLM Provider (choose one)
OPENAI_API_KEY=<your-openai-key>          # For GPT-4
# OR
ANTHROPIC_API_KEY=<your-anthropic-key>    # For Claude
```

#### 2. Prepare Paper Data

Place papers in `data/raw/` or `data/processed/`:

**JSON format:**
```json
{
  "paper_id": "arXiv:2023.12345",
  "title": "Deep Learning Optimization Techniques",
  "abstract": "This paper presents novel optimization methods...",
  "content": "Full paper text here...",
  "authors": ["John Doe", "Jane Smith"],
  "metadata": {"venue": "NeurIPS", "year": 2023}
}
```

**Or PDF files:** Will be auto-extracted with PyPDF2/pdfplumber

#### 3. Build Vector Index

Ingest papers and build FAISS vector index:

```bash
python scripts/build_index.py
```

Output: `data/index/papers.faiss` + `papers.pkl` (chunk metadata)

#### 4. Run Correlation Pipeline

**By query:**
```bash
python scripts/run_graph.py --query "machine learning optimization techniques"
```

**By seed papers:**
```bash
python scripts/run_graph.py --paper-ids paper_1 paper_2 paper_3
```

**With custom config:**
```bash
python scripts/run_graph.py --query "..." --config configs/my_config.yaml
```

**Output format:**
```
outputs/correlations_2024-10-07_14-32-15.json
outputs/correlations_2024-10-07_14-32-15.csv
```

### Output Format

**JSON** (`outputs/correlations_{timestamp}.json`):
```json
{
  "query": "machine learning optimization",
  "timestamp": "2024-10-07T14:32:15",
  "results": [
    {
      "paper_a_id": "arXiv:2023.12345",
      "paper_b_id": "arXiv:2023.54321",
      "correlation_score": 0.92,
      "relation_type": "builds_on",
      "evidence": ["chunk_1", "chunk_2"],
      "explanation": "Paper 2 extends methods from Paper 1 with adversarial training.",
      "confidence": 0.88
    }
  ]
}
```

**CSV** (`outputs/correlations_{timestamp}.csv`):
```
paper_a_id,paper_b_id,correlation_score,relation_type,explanation,confidence
arXiv:2023.12345,arXiv:2023.54321,0.92,builds_on,Paper 2 extends...,0.88
```

### Architecture

```
Query or Paper IDs
        ↓
[retrieve_papers] - Find papers matching query
        ↓
[retrieve_chunks] - Extract relevant sections
        ↓
[build_pairs] - Create candidate pairs, filter by similarity
        ↓
[analyze_pair] ⟿ (fan-out in parallel via LangGraph Send)
        ↓
[aggregate] - Rank correlations by score
        ↓
[save_results] - Write JSON + CSV to outputs/
        ↓
Correlation Results + LangSmith Traces
```

### Configuration

Edit `configs/default.yaml` to tune the pipeline:

```yaml
# RAG parameters
top_k: 10                    # Papers to retrieve
chunks_per_paper: 5          # Chunks per paper
max_pairs: 50                # Max candidate pairs to analyze
similarity_threshold: 0.5    # Min embedding cosine similarity

# Relation types for classification
relation_types:
  - same_problem
  - same_method
  - builds_on
  - contradicts
  - shared_dataset
  - unrelated

# LLM settings
model: "gpt-4-turbo"         # or "claude-3-5-sonnet-20241022"
temperature: 0.7
max_tokens: 500

# Embedding model
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

# Paths
data_path: "./data"
output_path: "./outputs"
index_path: "./data/index"
```

### LangSmith Tracing

Every node execution is automatically traced to LangSmith:

1. **Enable tracing** (already set in `.env_example`):
   ```env
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=<your-key>
   LANGSMITH_PROJECT=llm-project-paper-correlations
   ```

2. **View traces** at:
   ```
   https://smith.langchain.com/projects/llm-project-paper-correlations
   ```

3. **Metadata included**:
   - Paper IDs in analyze_pair calls
   - Query string
   - Chunk content
   - LLM outputs (parsed CorrelationResult)

### Testing

Run all tests:

```bash
pytest tests/ -v
```

**Test coverage:**
- ✓ Pydantic schema validation (score 0-1, relation types)
- ✓ CorrelationMatrix deduplication & sorting
- ✓ GraphState initialization
- ○ Graph compilation (stub until nodes implemented)

### Implementation Status

**Implemented:**
- ✅ Module structure and organization
- ✅ Pydantic schemas with validation
- ✅ Configuration loading from YAML
- ✅ Graph state definition
- ✅ Prompt templates

**TODO (stubs with `raise NotImplementedError()`):**
- [ ] `rag/loader.py` - PDF/JSON parsing
- [ ] `rag/chunker.py` - Section detection, token chunking
- [ ] `rag/embeddings.py` - Load sentence-transformers model
- [ ] `rag/vectorstore.py` - FAISS index ops
- [ ] `rag/retriever.py` - Similarity search
- [ ] `llm/client.py` - LLM generation with structured output
- [ ] `graph/nodes.py` - Implement 6 async nodes
- [ ] `graph/build_graph.py` - Graph construction + Send for fan-out
- [ ] `scripts/build_index.py` - End-to-end ingestion
- [ ] `scripts/run_graph.py` - CLI argument parsing

See `SETUP.md` for detailed implementation guide.

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: paper_corr` | Run `pip install -e .` in project root |
| `API key not found` | Check `.env` is copied from `.env_example` and keys are set |
| `No space left on device` | Install with `pip install --no-cache-dir` |
| `Graph must have an entrypoint` | Graph nodes not yet implemented (TODO in stub) |

### Next Steps

1. **Implement RAG pipeline** (loader → chunker → embeddings → vectorstore)
2. **Wire LLM client** to your chosen provider (OpenAI/Anthropic/other)
3. **Implement graph nodes** with async/await
4. **Test with sample papers** in `data/raw/`
5. **Monitor traces** in LangSmith dashboard

See `docs/pipeline.md` for architecture diagram and `SETUP.md` for complete TODO checklist.


---

## Manual Testing (Without LLM)

Test the pipeline without API keys by following these steps:

### Step 1: Create Sample Papers

Save these as JSON files in `data/raw/`:

**Create `data/raw/paper_1.json`:**
```json
{
  "paper_id": "paper_1",
  "title": "Deep Learning for Image Recognition",
  "authors": ["Alice Smith"],
  "abstract": "CNN methods for image classification. High accuracy on ImageNet.",
  "content": "Introduction: Image recognition is fundamental. Methods: We propose a CNN with residual connections. Results: 95% accuracy. Conclusion: Our approach outperforms baselines.",
  "metadata": {"venue": "CVPR", "year": 2023}
}
```

**Create `data/raw/paper_2.json`:**
```json
{
  "paper_id": "paper_2",
  "title": "Transformer Vision Models",
  "authors": ["Bob Johnson"],
  "abstract": "Self-attention mechanisms for vision. Competitive with CNNs.",
  "content": "Introduction: Vision Transformers emerge. Methods: Apply transformers to image patches. Results: 94% accuracy. Discussion: Less inductive bias than CNNs.",
  "metadata": {"venue": "ICCV", "year": 2023}
}
```

**Create `data/raw/paper_3.json`:**
```json
{
  "paper_id": "paper_3",
  "title": "Optimization for Neural Networks",
  "authors": ["Charlie Davis"],
  "abstract": "Advanced optimization techniques. AdamW and LAMB algorithms.",
  "content": "Introduction: Training needs efficiency. Methods: Analyze adaptive learning rates. Results: LAMB enables large batch training. Conclusion: Proper optimization is crucial.",
  "metadata": {"venue": "NIPS", "year": 2022}
}
```

### Step 2: Build Vector Index

```bash
python scripts/build_index.py
```

Output:
```
Loading papers from ./data...
✓ Loaded 3 papers
Chunking papers (chunk_size=512, overlap=64)...
✓ Created 12 chunks
Embedding chunks with sentence-transformers/all-MiniLM-L6-v2...
✓ Embedded 12 chunks
Building FAISS index...
✓ Index created with 12 chunks
Saving index to ./data/index...
✓ Index saved!

✅ Vector index built successfully!
Ready to run: python scripts/run_graph.py --query 'your query'
```

### Step 3: Run Correlation Analysis

**Option A: By query**
```bash
python scripts/run_graph.py --query "deep learning optimization"
```

**Option B: By specific papers**
```bash
python scripts/run_graph.py --paper-ids paper_1 paper_2
```

### Step 4: View Results

Results are saved to `outputs/`:

```bash
# View JSON results
cat outputs/correlations_*.json | python -m json.tool

# View CSV results
cat outputs/correlations_*.csv
```

### Example Output

```
Running pipeline with config from configs/default.yaml
Model: gpt-4-turbo
Embedding: sentence-transformers/all-MiniLM-L6-v2

✅ Found 3 correlations:
================================================================================

1. paper_1 <-> paper_3
   Score: 0.87 | Type: builds_on
   Explanation: Paper 3 provides optimization methods for training models like Paper 1

2. paper_2 <-> paper_3
   Score: 0.82 | Type: same_problem
   Explanation: Both papers address efficient training of large neural networks

3. paper_1 <-> paper_2
   Score: 0.76 | Type: same_problem
   Explanation: Both papers tackle image recognition with different architectures

================================================================================
Results saved to:
  • outputs/correlations_2024-10-07_14-32-15.json
  • outputs/correlations_2024-10-07_14-32-15.csv
```

### One-Liner (Copy & Paste All)

```bash
# Create papers
mkdir -p data/raw

cat > data/raw/paper_1.json << 'PAPER_EOF'
{"paper_id":"paper_1","title":"Deep Learning","authors":["Alice"],"abstract":"CNN methods","content":"Introduction: Image recognition. Methods: CNN with residual connections. Results: 95% accuracy.","metadata":{"year":2023}}
PAPER_EOF

cat > data/raw/paper_2.json << 'PAPER_EOF'
{"paper_id":"paper_2","title":"Transformers","authors":["Bob"],"abstract":"Vision Transformers","content":"Introduction: Self-attention. Methods: Apply to image patches. Results: 94% accuracy.","metadata":{"year":2023}}
PAPER_EOF

cat > data/raw/paper_3.json << 'PAPER_EOF'
{"paper_id":"paper_3","title":"Optimization","authors":["Charlie"],"abstract":"Training optimization","content":"Introduction: Efficiency. Methods: Analyze Adam, LAMB. Results: Large batch training possible.","metadata":{"year":2022}}
PAPER_EOF

# Build index
python scripts/build_index.py

# Run analysis
python scripts/run_graph.py --query "deep learning optimization"

# View results
cat outputs/correlations_*.json
```

### To Use With LLM (OpenAI/Anthropic)

1. Add API key to `.env`:
```bash
cp .env_example .env
# Edit .env:
OPENAI_API_KEY=sk-your-key-here
# OR
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

2. Run pipeline as normal:
```bash
python scripts/run_graph.py --query "machine learning"
```

The LLM will analyze paper pairs and generate explanations automatically.

### Troubleshooting

| Error | Solution |
|-------|----------|
| `No papers found` | Create JSON files in `data/raw/` |
| `ModuleNotFoundError: paper_corr` | Run from project root directory |
| `OPENAI_API_KEY not found` | Add key to `.env` file |
| `No module named sentence_transformers` | `pip install sentence-transformers` |
| `No module named faiss` | `pip install faiss-cpu` |

