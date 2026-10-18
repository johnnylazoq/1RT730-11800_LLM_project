"""Split documents into chunks for embedding."""
from dataclasses import dataclass


@dataclass
class TextChunk:
    """A single chunk of text with metadata."""
    text: str
    chunk_id: str
    source: str
    page_number: int | None = None
    start_char: int = 0
    end_char: int = 0


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
