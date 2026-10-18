"""Fetch papers from Semantic Scholar API."""
import os
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from app.llm.schemas import RetrievedChunk


def fetch_papers(query: str, limit: int = 10) -> list[dict]:
    """Fetch papers from Semantic Scholar API.
    
    Args:
        query: Search query (e.g., "vitamin C cold")
        limit: Maximum number of papers to fetch
        
    Returns:
        List of paper dictionaries with metadata
    """
    params = urlencode({
        "query": query,
        "limit": limit,
        "fields": "paperId,title,abstract,authors,year,url,externalIds,isOpenAccess,openAccessPdf"
    })
    headers = {"User-Agent": "scientific-claim-project/0.1"}
    if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
        headers["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
    
    request = Request(f"https://api.semanticscholar.org/graph/v1/paper/search?{params}", headers=headers)
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    
    papers = payload.get("data", [])
    return papers if isinstance(papers, list) else []


def papers_to_chunks(papers: list[dict]) -> list[RetrievedChunk]:
    """Convert papers to RetrievedChunk format.
    
    Uses abstract as the evidence chunk for each paper.
    """
    chunks = []
    for paper in papers:
        abstract = paper.get("abstract")
        paper_id = paper.get("paperId")
        
        if not abstract or not abstract.strip() or not paper_id:
            continue
        
        authors = [
            a["name"] if isinstance(a, dict) else a 
            for a in paper.get("authors", [])
        ]
        
        chunk = RetrievedChunk(
            paper_id=paper_id,
            chunk_id=f"{paper_id}:abstract",
            text=abstract,
            title=paper.get("title", "Untitled"),
            authors=authors,
            doi=(paper.get("externalIds") or {}).get("DOI"),
            url=paper.get("url"),
            year=paper.get("year"),
            score=paper.get("relevance_score"),
            metadata={
                "evidence_type": "abstract",
                "provider": "Semantic Scholar",
                "is_open_access": paper.get("isOpenAccess", False),
                "open_access_pdf": (paper.get("openAccessPdf") or {}).get("url")
            }
        )
        chunks.append(chunk)
    
    return chunks
