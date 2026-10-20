"""Compact public responses with separate timing and diagnostic artifacts."""
import json
from pathlib import Path
from time import perf_counter


def write_response(result, output, started_at):
    response = result.public_result()
    elapsed = round(perf_counter() - started_at, 3)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(response, indent=2), encoding='utf-8')
    path.with_suffix('.diagnostics.json').write_text(result.model_dump_json(indent=2), encoding='utf-8')
    path.with_suffix('.timing.json').write_text(json.dumps({
        'claim_to_response_seconds': elapsed,
        'scope': 'From claim acceptance by this command through final response generation; includes retrieval and ingestion when performed by this command, excludes trace upload and file writes.'
    }, indent=2), encoding='utf-8')
    print(json.dumps(response, indent=2), flush=True)
    print(f'Time from claim to final response: {elapsed:.3f} seconds', flush=True)
    return elapsed
