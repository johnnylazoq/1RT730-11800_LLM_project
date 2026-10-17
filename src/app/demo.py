"""A cached abstract adapter for testing before vector retrieval is available."""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from app.graph.workflow import verify_claim
from app.llm.schemas import RetrievedChunk


def fetch_once(query, output):
    """Exactly one API request, no retries or pagination. Never overwrite a cache."""
    path = Path(output)
    if path.exists():
        raise FileExistsError(f"Cache already exists: {path}")
    params = urlencode({"query": query, "limit": 5,
        "fields": "paperId,title,abstract,authors,year,url,externalIds"})
    headers = {"User-Agent": "scientific-claim-project/0.1"}
    if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
        headers["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
    request = Request("https://api.semanticscholar.org/graph/v1/paper/search?" + params, headers=headers)
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def abstract_chunks(payload):
    papers = payload if isinstance(payload, list) else payload.get("data", [])
    result = []
    for paper in papers:
        abstract = paper.get("abstract")
        paper_id = paper.get("paper_id") or paper.get("paperId")
        if not abstract or not abstract.strip() or not paper_id:
            continue
        authors = [a["name"] if isinstance(a, dict) else a for a in paper.get("authors", [])]
        result.append(RetrievedChunk(paper_id=paper_id, chunk_id=f"{paper_id}:abstract",
            text=abstract, title=paper.get("title") or "Untitled", authors=authors,
            doi=paper.get("doi") or (paper.get("externalIds") or {}).get("DOI"),
            url=paper.get("url"), year=paper.get("year"), score=paper.get("relevance_score"),
            metadata={"evidence_type": "abstract", "provider": "Semantic Scholar"}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--query", required=True)
    fetch.add_argument("--output", required=True)
    run = commands.add_parser("verify")
    run.add_argument("--claim", required=True)
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--format", choices=["abstracts", "chunks"], default="abstracts")
    run.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_once(args.query, args.output)
    else:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        chunks = abstract_chunks(payload) if args.format == "abstracts" else payload
        result = verify_claim(args.claim, chunks, max_workers=args.workers)
        Path(args.output).write_text(result.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
