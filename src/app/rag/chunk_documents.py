"""Split documents into chunks for embedding."""
from dataclasses import dataclass
import re
from app.rag.extract_documents import DocumentPage


@dataclass
class TextChunk:
    """A single chunk of text with metadata."""
    text: str
    chunk_id: str
    source: str
    page_number: int | None = None
    start_char: int = 0
    end_char: int = 0
    paragraph_number: int | None = None


def chunk_pages(pages: list[DocumentPage], source: str,
                strategy: str = "paragraph") -> list[TextChunk]:
    """One paragraph per chunk (or one page), never crossing page boundaries.

    Paragraph numbering restarts on each page. No character limits or overlap.
    A page without detected blank lines is one paragraph.
    """
    if strategy not in {"paragraph", "page"}:
        raise ValueError("strategy must be 'paragraph' or 'page'")
    chunks = []
    for page in pages:
        if page.page_number < 1:
            raise ValueError("Page numbers must be positive")
        parts = re.split(r"\n\s*\n", page.text.strip()) if strategy == "paragraph" else [page.text.strip()]
        parts = [part.strip() for part in parts if part.strip()]
        for number, text in enumerate(parts, start=1):
            suffix = f":paragraph_{number}" if strategy == "paragraph" else ""
            chunks.append(TextChunk(text=text,
                chunk_id=f"{source}:page_{page.page_number}{suffix}", source=source,
                page_number=page.page_number,
                paragraph_number=number if strategy == "paragraph" else None))
    return chunks


def chunk_text(text: str, source: str, chunk_size: int = 1000, overlap: int = 100) -> list[TextChunk]:
    """Split text into overlapping chunks.
    
    Args:
        text: Full text to chunk
        source: Source identifier (filename, paper_id, etc.)
        chunk_size: Characters per chunk
        overlap: Character overlap between chunks
        
    Returns:
        List of TextChunk objects
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    
    chunks = []
    step = chunk_size - overlap
    
    for i, start in enumerate(range(0, len(text), step)):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        
        if chunk_text.strip():
            chunk = TextChunk(
                text=chunk_text,
                chunk_id=f"{source}:chunk_{i}",
                source=source,
                start_char=start,
                end_char=end
            )
            chunks.append(chunk)
        
        if end >= len(text):
            break
    
    return chunks


def chunk_by_paragraphs(text: str, source: str, max_chunk_size: int = 2000) -> list[TextChunk]:
    """Split text into chunks respecting paragraph boundaries.
    
    Better for preserving semantic structure than simple character-based splitting.
    
    Args:
        text: Full text to chunk
        source: Source identifier
        max_chunk_size: Maximum chunk size in characters
        
    Returns:
        List of TextChunk objects
    """
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    chunk_idx = 0
    start_char = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if len(current_chunk) + len(para) + 2 <= max_chunk_size:
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += para
        else:
            if current_chunk:
                chunk = TextChunk(
                    text=current_chunk,
                    chunk_id=f"{source}:chunk_{chunk_idx}",
                    source=source,
                    start_char=start_char
                )
                chunks.append(chunk)
                chunk_idx += 1
                start_char += len(current_chunk)
            
            current_chunk = para
    
    if current_chunk:
        chunk = TextChunk(
            text=current_chunk,
            chunk_id=f"{source}:chunk_{chunk_idx}",
            source=source,
            start_char=start_char
        )
        chunks.append(chunk)
    
    return chunks
