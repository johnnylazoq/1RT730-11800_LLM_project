"""Load papers from data/ (PDF/JSON) into Paper objects."""

import json
from pathlib import Path
from typing import NamedTuple


class Paper(NamedTuple):
    """Paper metadata and content."""

    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    content: str  # Full text
    metadata: dict  # e.g., venue, year, doi


def load_papers(data_path: str | Path) -> list[Paper]:
    """Load papers from data directory (JSON format).

    Args:
        data_path: Path to data/raw or data/processed.

    Returns:
        List of Paper objects.
    """
    data_path = Path(data_path)
    papers = []

    # Try both raw and processed directories
    for json_file in data_path.glob("*.json"):
        try:
            with open(json_file) as f:
                paper_dict = json.load(f)
                paper = Paper(
                    paper_id=paper_dict.get("paper_id", json_file.stem),
                    title=paper_dict.get("title", ""),
                    authors=paper_dict.get("authors", []),
                    abstract=paper_dict.get("abstract", ""),
                    content=paper_dict.get("content", ""),
                    metadata=paper_dict.get("metadata", {}),
                )
                papers.append(paper)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return papers


def load_paper_by_id(paper_id: str, data_path: str | Path) -> Paper | None:
    """Load a single paper by ID.

    Args:
        paper_id: Paper identifier.
        data_path: Path to data directory.

    Returns:
        Paper object or None if not found.
    """
    data_path = Path(data_path)
    json_file = data_path / f"{paper_id}.json"

    if not json_file.exists():
        return None

    try:
        with open(json_file) as f:
            paper_dict = json.load(f)
            return Paper(
                paper_id=paper_dict.get("paper_id", paper_id),
                title=paper_dict.get("title", ""),
                authors=paper_dict.get("authors", []),
                abstract=paper_dict.get("abstract", ""),
                content=paper_dict.get("content", ""),
                metadata=paper_dict.get("metadata", {}),
            )
    except (json.JSONDecodeError, KeyError):
        return None
