"""Run a claims file sequentially, preserving per-claim logs and a batch summary."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--claims', type=Path, default=Path('scripts/claims.txt'))
    parser.add_argument('--queries', nargs='+', help='One search query per nonblank claim line')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--max-chunks-per-paper', type=int, default=3)
    parser.add_argument('--max-total-chunks', type=int, default=4)
    parser.add_argument('--max-chars-per-paper', type=int, default=8000)
    parser.add_argument('--langfuse', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('outputs/logs') / datetime.now().strftime('claims-%Y%m%d-%H%M%S'))
    args = parser.parse_args()
    claims = [line.strip() for line in args.claims.read_text().splitlines() if line.strip()]
    queries = args.queries or claims
    if not claims or len(queries) != len(claims):
        parser.error('Provide one query per claim, or omit --queries to search the claim text')
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    started = time.perf_counter()
    print(f'Batch artifacts: {args.output.resolve()}', flush=True)
    for index, (claim, query) in enumerate(zip(claims, queries), 1):
        destination = args.output / f'claim-{index:02d}'
        command = [sys.executable, str(Path(__file__).with_name('test_live_pipeline.py')),
            '--claim', claim, '--query', query, '--limit', str(args.limit),
            '--workers', str(args.workers), '--top-k', str(args.top_k),
            '--max-chunks-per-paper', str(args.max_chunks_per_paper),
            '--max-total-chunks', str(args.max_total_chunks),
            '--max-chars-per-paper', str(args.max_chars_per_paper), '--output', str(destination)]
        if args.langfuse:
            command.append('--langfuse')
        print(f'[{index}/{len(claims)}] Starting: {claim}', flush=True)
        claim_start = time.perf_counter()
        with (args.output / f'claim-{index:02d}.console.log').open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        metrics_path = destination / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {
            'claim': claim, 'query': query, 'status': 'failed_before_metrics',
            'workers': args.workers, 'paper_limit': args.limit}
        metrics.update(exit_code=completed.returncode, artifact_directory=str(destination),
                       process_seconds=round(time.perf_counter() - claim_start, 3))
        trace_file = destination / 'langfuse_trace.txt'
        if trace_file.exists():
            metrics['langfuse_url'] = trace_file.read_text().strip()
        results.append(metrics)
        summary = {'batch_elapsed_seconds': round(time.perf_counter() - started, 3), 'results': results}
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(f'[{index}/{len(claims)}] {metrics["status"]}: papers={metrics.get("papers_fetched", 0)}, '
              f'chunks={metrics.get("chunks_indexed", 0)}, workers={args.workers}, '
              f'seconds={metrics.get("claim_to_response_seconds", metrics["process_seconds"])}', flush=True)
    return int(any(result['exit_code'] != 0 for result in results))


if __name__ == '__main__':
    sys.exit(main())
