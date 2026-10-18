"""CLI for claim verification with optional RAG retrieval."""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from app.graph.workflow import verify_claim
from app.llm.schemas import RetrievedChunk
from app.rag.ingest import ingest_papers_from_api, ingest_pdfs
from app.rag.retriever import create_retriever


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
    
    # Original commands
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--query", required=True)
    fetch.add_argument("--output", required=True)
    
    verify = commands.add_parser("verify")
    verify.add_argument("--claim", required=True)
    verify.add_argument("--input", required=True)
    verify.add_argument("--output", required=True)
    verify.add_argument("--format", choices=["abstracts", "chunks"], default="abstracts")
    verify.add_argument("--workers", type=int, default=3)
    
    # RAG ingest commands
    ingest_api = commands.add_parser("ingest-api")
    ingest_api.add_argument("--query", required=True, help="Search query for papers")
    ingest_api.add_argument("--output", default="vector_db", help="Vector store output directory")
    ingest_api.add_argument("--limit", type=int, default=10, help="Max papers to fetch")
    
    ingest_pdf_cmd = commands.add_parser("ingest-pdf")
    ingest_pdf_cmd.add_argument("--input", required=True, help="Directory containing PDFs")
    ingest_pdf_cmd.add_argument("--output", default="vector_db", help="Vector store output directory")
    ingest_pdf_cmd.add_argument("--strategy", choices=["paragraph", "fixed"], default="paragraph")
    
    # RAG retrieval + verification
    rag_verify = commands.add_parser("rag-verify")
    rag_verify.add_argument("--claim", required=True, help="Claim to verify")
    rag_verify.add_argument("--vector-store", default="vector_db", help="Path to vector store")
    rag_verify.add_argument("--output", required=True, help="Output file for results")
    rag_verify.add_argument("--workers", type=int, default=3)
    rag_verify.add_argument("--top-k", type=int, default=10, help="Number of chunks to retrieve")
    
    args = parser.parse_args()
    
    if args.command == "fetch":
        fetch_once(args.query, args.output)
        print(f"Saved to {args.output}")
    
    elif args.command == "verify":
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        chunks = abstract_chunks(payload) if args.format == "abstracts" else payload
        result = verify_claim(args.claim, chunks, max_workers=args.workers)
        Path(args.output).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"Saved to {args.output}")
    
    elif args.command == "ingest-api":
        ingest_papers_from_api(
            query=args.query,
            vector_store_path=args.output,
            limit=args.limit
        )
        print(f"Vector store created at {args.output}")
    
    elif args.command == "ingest-pdf":
        ingest_pdfs(
            pdf_dir=args.input,
            vector_store_path=args.output,
            chunk_strategy=args.strategy
        )
        print(f"Vector store created at {args.output}")
    
    elif args.command == "rag-verify":
        retriever = create_retriever(args.vector_store)
        chunks = retriever.retrieve(args.claim, k=args.top_k)
        
        if not chunks:
            print(f"No relevant chunks found for: {args.claim}")
            result = None
        else:
            print(f"Retrieved {len(chunks)} relevant chunks")
            result = verify_claim(args.claim, chunks, max_workers=args.workers)
        
        output_data = {
            "claim": args.claim,
            "result": result.model_dump() if result else None,
            "retrieved_chunks_count": len(chunks)
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2), encoding="utf-8")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
