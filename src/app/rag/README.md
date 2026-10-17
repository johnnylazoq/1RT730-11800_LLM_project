# RAG implementation handoff

The Python files in this folder and `../graph/nodes/rag.py` are intentionally
empty placeholders for the colleague implementing upstream RAG. No RAG behavior
is implemented or connected to the existing graph yet.

## File responsibilities

| File | Intended responsibility |
| --- | --- |
| `collect_papers.py` | Collect paper metadata and available documents. Reuse or refactor the existing Semantic Scholar retrieval script rather than duplicate it. |
| `extract_documents.py` | Extract text from collected documents, preserving page numbers and source metadata. |
| `chunk_documents.py` | Split extracted text into chunks with stable paper and chunk IDs and page/section provenance. |
| `embeddings.py` | Generate document and query embeddings using the same model and configuration. |
| `vector_store.py` | Store and search embeddings together with original chunk text and metadata; support repeat ingestion without duplicate records. |
| `ingest.py` | Coordinate the offline collection, extraction, chunking, embedding, and storage flow. |
| `retriever.py` | Accept a claim, embed the query, search the store, and return retrieved chunk records. |
| `../graph/nodes/rag.py` | Adapt retrieval to graph state: read `claim`, produce `retrieved_chunks`, and eventually connect before `group_papers`. |
| `__init__.py` | Package marker; intentionally empty. |

## Planned flow

Offline: collect papers → extract documents → chunk text → embed → store.

Online: claim → RAG node → rank/group papers → parallel paper analysis →
aggregate evidence and validate citations → structured response.

Ranking/grouping and every subsequent online step already exist. Reuse them.
The current workflow still starts at grouping and accepts supplied chunks, so
the demo and downstream tests can run independently of RAG development.

## Integration contract

Use `RetrievedChunk` from `app.llm.schemas`; do not introduce a competing chunk
model at the downstream boundary.

- Required fields: `paper_id`, `chunk_id`, `text`, and `title`.
- Optional fields: `authors`, `doi`, `url`, `year`, `page_number`, `score`, and
  the `metadata` dictionary for additional provenance.
- Preserve exact excerpt text and stable IDs. Chunk IDs are unique within a
  paper; paper IDs identify the same paper across retrieval results.
- Keep bibliographic metadata consistent across chunks from the same paper.
- A PDF URL without extracted text is not an evidence chunk. Identify abstract
  evidence with `metadata.evidence_type = "abstract"`.
- Document score direction. Existing grouping defaults to higher scores first;
  configure `higher_score_better=False` if returning distances. Missing scores
  are supported and preserve relative retrieval order.
- Populate `evidence_cutoff` only when the corpus has a known coverage cutoff;
  a retrieval timestamp is not an evidence cutoff.

Until graph integration is ready, pass retrieval results to the existing
`verify_claim(claim, retrieved_chunks, ...)` entry point in
`app.graph.workflow`. Later, connect the RAG node between START and
`group_papers`, preserving a way to supply chunks directly for tests.

## Existing functionality to consult

- `scripts/Semantic Scholar retrieval.py`: existing paper search and metadata mapping.
- `scripts/relevance_filter.py`: existing experimental paper relevance scoring.
- `app.demo`: one-request Semantic Scholar fetch and cached abstract adapter.
- `app.llm.schemas`: shared evidence and final response models.
- `app.graph.nodes.group_papers`: existing ranking, deduplication, and grouping.
- `tests/test_pipeline.py`: downstream contract and behavior tests.

The colleague should add tests for extraction provenance, stable chunk IDs,
embedding/store compatibility, repeat ingestion, empty retrieval, and the
retriever-to-grouping contract when implementing these components. No provider,
embedding model, database, or new dependency has been selected by this scaffold.
