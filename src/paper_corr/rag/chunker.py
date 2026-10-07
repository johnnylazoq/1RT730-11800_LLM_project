"""Split papers into chunks with section labels."""

from typing import NamedTuple

from paper_corr.rag.loader import Paper


class Chunk(NamedTuple):
    """A chunk of a paper with section label."""

    chunk_id: str  # e.g., "paper_123_section_2_chunk_0"
    paper_id: str
    section: str  # e.g., "abstract", "introduction", "methods"
    text: str
    start_idx: int  # Character index in original paper
    end_idx: int


def chunk_paper(paper: Paper, chunk_size: int = 512, overlap: int = 64) -> list[Chunk]:
    """Split paper into chunks by section and sliding window.

    Args:
        paper: Paper object.
        chunk_size: Target chunk size in characters.
        overlap: Character overlap between chunks.

    Returns:
        List of Chunk objects.
    """
    chunks = []
    chunk_counter = 0

    # Simple section detection (can be enhanced with regex)
    sections = [
        ("abstract", paper.abstract),
        ("content", paper.content),
    ]

    for section_name, section_text in sections:
        if not section_text:
            continue

        # Sliding window chunking
        for i in range(0, len(section_text), chunk_size - overlap):
            chunk_text = section_text[i : i + chunk_size]
            if len(chunk_text.strip()) < 50:  # Skip tiny chunks
                continue

            chunk = Chunk(
                chunk_id=f"{paper.paper_id}_{section_name}_{chunk_counter}",
                paper_id=paper.paper_id,
                section=section_name,
                text=chunk_text,
                start_idx=i,
                end_idx=min(i + chunk_size, len(section_text)),
            )
            chunks.append(chunk)
            chunk_counter += 1

    return chunks


def chunk_papers(papers: list[Paper], chunk_size: int = 512, overlap: int = 64) -> list[Chunk]:
    """Chunk multiple papers.

    Args:
        papers: List of Paper objects.
        chunk_size: Target chunk size in characters.
        overlap: Character overlap between chunks.

    Returns:
        List of Chunk objects.
    """
    all_chunks = []
    for paper in papers:
        all_chunks.extend(chunk_paper(paper, chunk_size, overlap))
    return all_chunks
