"""Compare one/two workers on fresh coffee and vitamin-D searches; no paper reuse."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--claims', type=Path, default=Path('scripts/claims.txt'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    claims = args.claims.read_text().splitlines()
    if len(claims) < 4:
        parser.error('Expected coffee and vitamin D claims on lines 3 and 4')
    trials = [(2, 1), (2, 2), (3, 2), (3, 1)]
    queries = {2: 'coffee cardiovascular disease', 3: 'vitamin D supplementation depression randomized trial'}
    results = []
    for claim_index, workers in trials:
        name = f'claim-{claim_index + 1:02d}-workers-{workers}'
        destination = args.output / name
        # Equalize cold model startup. Evidence is fetched/processed afresh each trial.
        warmup = httpx.post(os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434') + '/api/generate',
            json={'model': os.getenv('OLLAMA_MODEL', 'frob/qwen3.5-instruct:9b'),
                  'prompt': 'Ready', 'stream': False, 'keep_alive': '10m',
                  'options': {'num_predict': 1, 'num_ctx': 16384}}, timeout=240)
        warmup.raise_for_status()
        command = [sys.executable, str(Path(__file__).with_name('test_live_pipeline.py')),
            '--query', queries[claim_index], '--claim', claims[claim_index],
            '--limit', '20', '--top-k', '4', '--max-total-chunks', '4',
            '--workers', str(workers), '--langfuse', '--output', str(destination)]
        print(f'Starting {name}', flush=True)
        started = time.perf_counter()
        with (args.output / f'{name}.console.log').open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        metrics_path = destination / 'metrics.json'
        result = json.loads(metrics_path.read_text()) if metrics_path.exists() else {'status': 'failed_before_metrics'}
        result.update(trial=name, exit_code=completed.returncode, process_seconds=round(time.perf_counter()-started, 3))
        trace_path = destination / 'trace.jsonl'
        events = [json.loads(line) for line in trace_path.read_text().splitlines()] if trace_path.exists() else []
        llm_tokens = 0
        paper_tokens = []
        for event in events:
            if event['stage'] == 'retrieval' and event['event'] == 'output':
                evidence = [(c['paper_id'], c['chunk_id'], c['text']) for c in event['chunks']]
                result['retrieved_evidence_sha256'] = hashlib.sha256(json.dumps(evidence).encode()).hexdigest()
            if event['stage'] == 'claim_verification' and event['event'] == 'end':
                result['verification_seconds'] = round(event['seconds'], 3)
            if event['stage'] == 'llm' and event['event'] == 'end':
                tokens = event['response']['generations'][0][0].get('generation_info', {}).get('eval_count', 0)
                llm_tokens += tokens
                paper_tokens.append(tokens)
        result['llm_output_tokens'] = llm_tokens
        result['paper_output_tokens'] = paper_tokens[:-1]
        results.append(result)
        (args.output / 'benchmark.json').write_text(json.dumps({'trials': results}, indent=2))
        print(f'{name}: {result["status"]}, total={result.get("claim_to_response_seconds")}, '
              f'verification={result.get("verification_seconds")}, output_tokens={llm_tokens}', flush=True)
    return int(any(r['exit_code'] != 0 for r in results))


if __name__ == '__main__':
    sys.exit(main())
