"""CLI to run the paper correlation graph."""

import argparse
import json
import sys
from pathlib import Path

# Add src/ to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from paper_corr.config import load_config
from paper_corr.graph.build_graph import run_pipeline


def main():
    """Parse arguments and run pipeline."""
    parser = argparse.ArgumentParser(description="Run paper correlation pipeline")
    parser.add_argument("--query", type=str, help="Query string")
    parser.add_argument(
        "--paper-ids",
        type=str,
        nargs="+",
        help="Seed paper IDs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Config file",
    )

    args = parser.parse_args()

    if not args.query and not args.paper_ids:
        parser.error("Provide either --query or --paper-ids")

    config = load_config(args.config)
    print(f"Running pipeline with config from {args.config}")
    print(f"Model: {config.get('model')}")
    print(f"Embedding: {config.get('embedding_model')}")

    # Run pipeline
    query = args.query or " ".join(args.paper_ids)
    result = run_pipeline(query=query, config_path=args.config, seed_paper_ids=args.paper_ids or [])

    # Print results
    correlations = result.get("correlations", [])
    print(f"\n✅ Found {len(correlations)} correlations:")
    print("=" * 80)

    for i, corr in enumerate(correlations[:10], 1):  # Show top 10
        print(f"\n{i}. {corr.paper_a_id} <-> {corr.paper_b_id}")
        print(f"   Score: {corr.correlation_score:.2f} | Type: {corr.relation_type}")
        print(f"   Explanation: {corr.explanation}")

    # Print output file info
    output_files = result.get("output_files", [])
    if output_files:
        print("\n" + "=" * 80)
        print("Results saved to:")
        for f in output_files:
            print(f"  • {f}")


if __name__ == "__main__":
    main()
