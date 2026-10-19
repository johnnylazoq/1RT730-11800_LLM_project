"""Semantic Scholar discovery and verification of linked open-access PDFs."""
import json
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from app.llm.schemas import RetrievedChunk
from app.rag.chunk_documents import chunk_pages
from app.rag.extract_documents import extract_pages_from_pdf

USER_AGENT = "scientific-claim-project/0.1"
FIELDS = "paperId,title,abstract,authors,year,url,externalIds,isOpenAccess,openAccessPdf"
MAX_PDF_BYTES = 30 * 1024 * 1024


def fetch_papers(query: str, limit: int = 10) -> list[dict]:
    """Fetch relevance-ranked metadata, paging in batches of up to 100."""
    if not query.strip() or not 1 <= limit <= 1000:
        raise ValueError("Provide a nonblank query and a limit between 1 and 1000")
    papers = []
    seen = set()
    offset = 0
    while len(papers) < limit and offset < 1000:
        params = urlencode({"query": query, "limit": min(100, limit - len(papers)),
                            "offset": offset, "fields": FIELDS})
        headers = {"User-Agent": USER_AGENT}
        if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
            headers["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
        request = Request(f"https://api.semanticscholar.org/graph/v1/paper/search?{params}", headers=headers)
        for attempt in range(3):
            try:
                with urlopen(request, timeout=30) as response:
                    payload = json.load(response)
                break
            except HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise
                delay = exc.headers.get("Retry-After", "") if exc.headers else ""
                time.sleep(min(float(delay), 30) if delay.isdigit() else 2 ** attempt)
        data = payload.get("data", [])
        if not isinstance(data, list):
            raise ValueError("Semantic Scholar returned invalid paper data")
        for paper in data:
            paper_id = paper.get("paperId")
            if paper_id and paper_id not in seen:
                seen.add(paper_id)
                papers.append(paper)
        next_offset = payload.get("next")
        if not data or not isinstance(next_offset, int) or next_offset <= offset:
            break
        offset = next_offset
    return papers[:limit]


def _bibliography(paper: dict) -> dict:
    return {
        "paper_id": paper["paperId"], "title": paper.get("title") or "Untitled",
        "authors": [a["name"] if isinstance(a, dict) else a for a in (paper.get("authors") or [])],
        "doi": (paper.get("externalIds") or {}).get("DOI"),
        "url": paper.get("url"), "year": paper.get("year"),
        "score": paper.get("relevance_score"),
    }


def papers_to_chunks(papers: list[dict]) -> list[RetrievedChunk]:
    """Offline abstract adapter; does not claim to have checked full text."""
    return [RetrievedChunk(**_bibliography(paper),
                chunk_id=f"{paper['paperId']}:abstract", text=paper["abstract"],
                metadata={"evidence_type": "abstract", "provider": "Semantic Scholar",
                          "full_text_status": "not_checked",
                          "open_access_pdf": (paper.get("openAccessPdf") or {}).get("url")})
            for paper in papers if paper.get("paperId") and (paper.get("abstract") or "").strip()]


def resolve_paper(paper: dict, *, strategy: str = "paragraph",
                  abstract_fallback: bool = True) -> tuple[list[RetrievedChunk], dict]:
    """Check the linked PDF and return page-aware evidence plus an audit record.

    An absent/broken PDF link does not prove no full text exists elsewhere.
    Non-PDF landing pages are reported, never ingested as paper evidence.
    """
    if strategy not in {"paragraph", "page"}:
        raise ValueError("strategy must be 'paragraph' or 'page'")
    bibliography = _bibliography(paper)
    pdf_url = (paper.get("openAccessPdf") or {}).get("url")
    pdf_url_source = "openAccessPdf" if pdf_url else None
    arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
    if not pdf_url and arxiv_id and re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-zA-Z.-]+/\d{7})(?:v\d+)?", arxiv_id):
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        pdf_url_source = "externalIds.ArXiv"
    report = {"paper_id": paper["paperId"], "title": bibliography["title"],
              "pdf_url": pdf_url, "pdf_url_source": pdf_url_source,
              "is_open_access": paper.get("isOpenAccess", False),
              "full_text_status": "no_pdf_url", "document_format": None,
              "page_count": 0, "pages_with_text": 0, "empty_pages": []}
    pages = []
    if pdf_url:
        if urlsplit(pdf_url).scheme not in {"https", "http"}:
            report["full_text_status"] = "invalid_pdf_url"
        else:
            try:
                # Never send the Semantic Scholar API key to a document host.
                request = Request(pdf_url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf"})
                with urlopen(request, timeout=30) as response:
                    report["resolved_url"] = response.geturl()
                    report["content_type"] = response.headers.get("Content-Type", "")
                    content = response.read(MAX_PDF_BYTES + 1)
                if len(content) > MAX_PDF_BYTES:
                    report["full_text_status"] = "download_too_large"
                elif b"%PDF-" not in content[:1024]:
                    report["full_text_status"] = "non_pdf_response"
                    report["document_format"] = "non_pdf"
                else:
                    report["document_format"] = "pdf"
                    try:
                        pages = extract_pages_from_pdf(content)
                    except Exception as exc:
                        report["full_text_status"] = "pdf_extraction_failed"
                        report["error_type"] = type(exc).__name__
                    else:
                        report["page_count"] = len(pages)
                        report["pages_with_text"] = sum(bool(p.text.strip()) for p in pages)
                        report["empty_pages"] = [p.page_number for p in pages if not p.text.strip()]
                        report["full_text_status"] = ("pdf_text_available" if report["pages_with_text"]
                                                       else "pdf_without_extractable_text")
            except (URLError, TimeoutError, OSError, ValueError) as exc:
                report["full_text_status"] = "download_failed"
                report["error_type"] = type(exc).__name__
                if isinstance(exc, HTTPError):
                    report["http_status"] = exc.code
    chunks = [RetrievedChunk(**bibliography, chunk_id=c.chunk_id, text=c.text,
                  page_number=c.page_number,
                  metadata={**report, "provider": "Semantic Scholar", "evidence_type": "pdf_chunk",
                            "paragraph_number": c.paragraph_number})
              for c in chunk_pages(pages, paper["paperId"], strategy)]
    if not chunks and abstract_fallback:
        chunks = papers_to_chunks([paper])
        for chunk in chunks:
            chunk.metadata.update(report)
    report["evidence_type"] = chunks[0].metadata["evidence_type"] if chunks else "none"
    report["chunk_count"] = len(chunks)
    return chunks, report
