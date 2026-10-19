"""Extract PDF pages without losing physical page numbering."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re


@dataclass(frozen=True)
class DocumentPage:
    page_number: int
    text: str


def extract_pages_from_pdf(pdf: str | Path | bytes) -> list[DocumentPage]:
    """Keep blank pages to preserve 1-based physical page numbers.

    Layout extraction retains vertical gaps for heuristic paragraph boundaries.
    No OCR is performed.
    """
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(pdf) if isinstance(pdf, bytes) else pdf)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        # PyPDF layout mode raises on truly blank pages without a content stream.
        text = page.extract_text(extraction_mode="layout") if page.get("/Contents") is not None else ""
        pages.append(DocumentPage(number, _clean_text(text or "")))
    return pages


def extract_text_from_pdf(pdf_path: str) -> str:
    """Compatibility helper; ingestion uses extract_pages_from_pdf instead."""
    return "\n\n".join(page.text for page in extract_pages_from_pdf(pdf_path))


def extract_from_multiple_pdfs(pdf_dir: str) -> dict[str, str]:
    return {path.name: extract_text_from_pdf(str(path))
            for path in sorted(Path(pdf_dir).glob("*.pdf"))}


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
