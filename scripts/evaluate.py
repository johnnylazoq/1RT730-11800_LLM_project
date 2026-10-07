"""Evaluate correlation predictions against ground truth (if available)."""

import sys
from pathlib import Path

# Add src/ to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from paper_corr.config import load_config


def main():
    """Evaluate predictions.

    TODO: If paper citations or gold labels are available:
    1. Load predictions from outputs/
    2. Load ground truth
    3. Calculate precision, recall, F1
    4. Save metrics
    """
    config = load_config()
    print("Evaluation script stub")

    # TODO: Implement
    raise NotImplementedError("evaluate.py not yet implemented")


if __name__ == "__main__":
    main()
