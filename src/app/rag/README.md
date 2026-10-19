# Backend paper ingestion and retrieval

Semantic Scholar search returns bibliographic metadata, abstracts, and an optional
`openAccessPdf.url`; it does not return the PDF body. `ingest-api` follows that
link (or the official arXiv PDF when an arXiv ID is available but no PDF link is
provided), checks the response bytes for a PDF header, and extracts its text. Neither
`isOpenAccess` nor a URL by itself counts as verified readable evidence.

Each searched paper is recorded in `ingestion_report.json`, including its PDF URL,
resolved URL when available, document format, extraction status, page counts,
empty pages, evidence type, and chunk count. Statuses distinguish missing links,
failed downloads, non-PDF responses, invalid PDFs, PDFs without extractable text,
and PDFs with text. Missing or failed links do not prove full text is unavailable
elsewhere. HTML landing pages are reported but are not scraped as paper evidence.
Downloads have a 30-second timeout and a 30 MiB size limit.

## Splitting and provenance

- Default `--strategy paragraph`: one extracted paragraph per chunk, within a page.
- `--strategy page`: one nonempty PDF page per chunk.
- No character-size splitting, merging, or overlap in either ingestion strategy.
- `page_number` is the 1-based physical PDF page, not a printed page label.
- Blank pages produce no chunks but do not renumber subsequent pages.
- Paragraph numbers restart at 1 per page; IDs are
  `<paper_id>:page_<n>:paragraph_<m>` (or `<paper_id>:page_<n>` for page mode).
- Paragraphs are inferred from blank lines in PyPDF layout extraction. If none
  are detected, the entire page is one paragraph. Multicolumn layouts, tables,
  headers, and footers can affect extraction order and boundaries.
- Scanned/image-only PDFs need OCR, which is not implemented here.
- Text, page numbers, paragraph numbers, and evidence type survive FAISS metadata
  persistence, retrieval, grouping, and final source citations.

API ingestion prefers extracted PDF evidence. If unavailable, it uses an explicitly
labeled abstract with no page number. `--full-text-only` disables that fallback.
If nothing is usable, ingestion raises an error after saving the report.
The `fetch`/`verify --format abstracts` commands remain the legacy abstract route.

## Run

Use Python 3.12 for the declared dependency set (including NumPy <2):

```bash
python3.12 -m venv .venv312
.venv312/bin/python -m pip install -e '.[test]'

.venv312/bin/python -m app.demo ingest-api \
  --query "your research topic" --limit 5 \
  --strategy paragraph --output vector_db_papers

.venv312/bin/python -m app.demo ingest-pdf \
  --input /path/to/pdfs --strategy paragraph --output vector_db_local

.venv312/bin/python -m app.demo rag-verify \
  --claim "Your claim" --vector-store vector_db_papers \
  --output /tmp/paper-verdict.json --top-k 10

.venv312/bin/python -m pytest \
  tests/test_rag.py tests/test_pipeline.py tests/test_fulltext_pipeline.py -v
```

Set `SEMANTIC_SCHOLAR_API_KEY` if available. API rate limits can still prevent a
live search; transient HTTP errors have bounded retries. Verification requires
Ollama; unit/integration workflow tests use a fake LLM. The embedding test uses
MiniLM and may download its weights. Exclude it with `-k 'not test_embedding_model'`
for tests without model downloads. No UI tests are included in `tests/`.

Ingestion builds a new index at its output path; use a fresh directory for a new
corpus. Existing indexes must be rebuilt to obtain page-aware chunks. Retrieval
selects chunks, not complete papers; the downstream paper/chunk/context budgets
still apply. MiniLM has a finite token window, so long paragraphs/pages may be
truncated for embedding even though the stored text remains intact.

On the tested macOS environment, initializing FAISS before importing PyTorch can
abort with a duplicate OpenMP runtime error. The explicit test order above loads
the embedding model first, as the production ingestion/retriever constructors do.
Alternatively, run the embedding test and the remaining tests in separate Python
processes. Do not suppress the runtime error with `KMP_DUPLICATE_LIB_OK`.
