"""Normalize the retrieval boundary without performing retrieval or embedding."""
from app.llm.schemas import EvidenceChunk, RetrievedChunk, ScientificPaper


def group_chunks(records: list[RetrievedChunk], *, higher_score_better: bool = True,
                 max_papers: int = 10, max_chunks_per_paper: int = 8,
                 max_chars_per_paper: int = 24000) -> tuple[list[ScientificPaper], list[str]]:
    if min(max_papers, max_chunks_per_paper, max_chars_per_paper) < 1:
        raise ValueError("Evidence limits must be positive")
    papers: dict[str, ScientificPaper] = {}
    seen: dict[tuple[str, str], RetrievedChunk] = {}
    notes = []
    for record in records:
        if not record.text.strip() or not record.paper_id.strip() or not record.chunk_id.strip():
            raise ValueError("Chunk text and identifiers must not be blank")
        key = (record.paper_id, record.chunk_id)
        if key in seen:
            if record != seen[key]:
                raise ValueError(f"Conflicting duplicate chunk: {key}")
            continue
        seen[key] = record
        metadata = record.model_dump(include={"paper_id", "title", "authors", "doi", "url", "year"})
        if record.paper_id not in papers:
            papers[record.paper_id] = ScientificPaper(**metadata)
        paper = papers[record.paper_id]
        for field, value in metadata.items():
            old = getattr(paper, field)
            if old and value and old != value:
                raise ValueError(f"Conflicting paper metadata: {record.paper_id}/{field}")
            if not old and value:
                setattr(paper, field, value)
        paper.chunks.append(EvidenceChunk(**record.model_dump()))

    def rank(chunk):
        return (chunk.score is None, (-chunk.score if higher_score_better else chunk.score)
                if chunk.score is not None else 0)

    for paper in papers.values():
        paper.chunks.sort(key=rank)
    ordered = sorted(papers.values(), key=lambda paper: rank(paper.chunks[0]))
    if len(ordered) > max_papers:
        notes.append(f"Paper limit omitted {len(ordered) - max_papers} retrieved papers.")
    selected = []
    for paper in ordered[:max_papers]:
        kept = []
        used = 0
        for chunk in paper.chunks:
            if len(kept) < max_chunks_per_paper and used + len(chunk.text) <= max_chars_per_paper:
                kept.append(chunk)
                used += len(chunk.text)
        if len(kept) != len(paper.chunks):
            notes.append(f"Evidence limits omitted {len(paper.chunks) - len(kept)} chunks from {paper.paper_id}.")
        selected.append(paper.model_copy(update={"chunks": kept}))
    return selected, notes
