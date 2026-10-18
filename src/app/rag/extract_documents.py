"""Extract text from PDF documents."""
import re
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file.
    
    Uses PyPDF for extraction. Falls back gracefully if pypdf is not available.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("PyPDF not installed. Run: pip install pypdf")
    
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    return _clean_text(text)


def extract_from_multiple_pdfs(pdf_dir: str) -> dict[str, str]:
    """Extract text from all PDFs in a directory.
    
    Args:
        pdf_dir: Directory containing PDF files
        
    Returns:
        Dictionary mapping filename to extracted text
    """
    pdf_dir = Path(pdf_dir)
    results = {}
    
    for pdf_file in pdf_dir.glob("*.pdf"):
        try:
            results[pdf_file.name] = extract_text_from_pdf(str(pdf_file))
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
    
    return results


def _clean_text(text: str) -> str:
    """Clean extracted text: remove excessive whitespace, normalize line breaks."""
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove excessive spaces
    text = re.sub(r' {2,}', ' ', text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()
