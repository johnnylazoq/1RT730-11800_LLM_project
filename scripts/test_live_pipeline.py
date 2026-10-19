"""Run the real pipeline and retain local stage/LLM traces (no cloud tracing required)."""
import argparse
from contextlib import contextmanager, nullcontext, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
import time

from langchain_core.callbacks import BaseCallbackHandler
from app.graph.workflow import build_graph
from app.reporting import write_response
from app.llm.client import get_llm
from app.rag.collect_papers import fetch_papers, resolve_paper
from app.rag.ingest import ingest_chunks
from app.rag.retriever import create_retriever
from app.rag.embeddings import EmbeddingModel


class Trace:
    def __init__(self, path, cloud=None):
        self.path = path
        self.cloud = cloud
        self.callbacks = []
        self.lock = threading.Lock()

    def redact(self, text):
        for name, value in os.environ.items():
            if value and any(word in name.upper() for word in ('KEY', 'TOKEN', 'PASSWORD', 'SECRET')):
                text = text.replace(value, '[REDACTED]')
        return text

    def event(self, stage, event, **data):
        record = dict(time=datetime.now(timezone.utc).isoformat(), stage=stage, event=event, **data)
        line = self.redact(json.dumps(record, default=lambda x: x.model_dump() if hasattr(x, 'model_dump') else str(x)))
        with self.lock:
            with self.path.open('a', encoding='utf-8') as handle:
                handle.write(line + '\n')
        if self.cloud and event in {'output', 'passed'}:
            self.cloud.update_current_span(output=json.loads(line))
        print(f"[{record['time']}] {stage}: {event}", flush=True)

    @contextmanager
    def stage(self, name):
        start = time.monotonic()
        self.event(name, 'start')
        observation = self.cloud.start_as_current_observation(name=name, as_type='span') if self.cloud else nullcontext()
        with observation:
            try:
                yield
            except Exception as exc:
                self.event(name, 'error', error_type=type(exc).__name__, message=str(exc), seconds=time.monotonic() - start)
                raise
            else:
                self.event(name, 'end', seconds=time.monotonic() - start)


class Tee:
    def __init__(self, terminal, log, trace):
        self.terminal, self.log, self.trace = terminal, log, trace
        self.lock = threading.Lock()

    def write(self, text):
        text = self.trace.redact(text)
        with self.lock:
            self.terminal.write(text)
            self.log.write(text)
            self.log.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class LLMTrace(BaseCallbackHandler):
    def __init__(self, trace):
        self.trace = trace

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.trace.event('llm', 'start', run_id=str(run_id), messages=[[m.model_dump() for m in batch] for batch in messages])

    def on_llm_end(self, response, *, run_id, **kwargs):
        self.trace.event('llm', 'end', run_id=str(run_id), response=response.model_dump())

    def on_llm_error(self, error, *, run_id, **kwargs):
        self.trace.event('llm', 'error', run_id=str(run_id), message=str(error))


def run(args, trace):
    if not os.getenv('SEMANTIC_SCHOLAR_API_KEY'):
        raise ValueError('Set SEMANTIC_SCHOLAR_API_KEY before running')
    trace.event('configuration', 'info', query=args.query, claim=args.claim, limit=args.limit,
                workers=args.workers, top_k=args.top_k,
                max_chunks_per_paper=args.max_chunks_per_paper, max_total_chunks=args.max_total_chunks,
                model=os.getenv('OLLAMA_MODEL', 'frob/qwen3.5-instruct:9b'))
    print(f'Configuration: paper limit={args.limit}, top_k={args.top_k}, workers={args.workers}', flush=True)
    with trace.stage('semantic_scholar_search'):
        papers = fetch_papers(args.query, limit=args.limit)
        args.metrics['papers_fetched'] = len(papers)
        trace.event('semantic_scholar_search', 'output', papers=papers)
    chunks, reports = [], []
    for paper in papers:
        with trace.stage('pdf_download_extract_chunk:' + paper['paperId']):
            evidence, report = resolve_paper(paper)
            chunks.extend(evidence)
            reports.append(report)
            args.metrics['chunks_indexed'] = len(chunks)
            args.metrics['papers_with_evidence'] = sum(r['chunk_count'] > 0 for r in reports)
            trace.event('pdf_download_extract_chunk', 'output', report=report, chunks=evidence)
    (args.output / 'ingestion_report.json').write_text(json.dumps(reports, indent=2))
    with trace.stage('embedding_and_vector_storage'):
        embedding_model = EmbeddingModel()
        ingest_chunks(chunks, str(args.output / 'vector_store'), embedding_model=embedding_model)
    with trace.stage('reload_and_retrieval'):
        retriever = create_retriever(str(args.output / 'vector_store'), embedding_model=embedding_model)
        retrieved = retriever.retrieve(args.claim, k=args.top_k)
        args.metrics['chunks_retrieved'] = len(retrieved)
        trace.event('retrieval', 'output', chunks=retrieved)
        if not retrieved:
            raise ValueError('Live test retrieved no evidence')
    llm = get_llm()
    llm.callbacks = [LLMTrace(trace)]
    graph = build_graph(llm=llm, max_workers=args.workers, retries=1,
        max_chunks_per_paper=args.max_chunks_per_paper, max_total_chunks=args.max_total_chunks,
        max_chars_per_paper=args.max_chars_per_paper)
    final = None
    with trace.stage('claim_verification'):
        for update in graph.stream({'claim': args.claim, 'retrieved_chunks': retrieved, 'evidence_cutoff': None},
                                  config={'callbacks': trace.callbacks}, stream_mode='updates'):
            for node, output in update.items():
                if node == 'group_papers':
                    args.metrics['papers_analyzed'] = len(output['papers'])
                    args.metrics['chunks_used_for_analysis'] = sum(len(p.chunks) for p in output['papers'])
                    args.metrics['effective_worker_limit'] = min(args.workers, len(output['papers']))
                trace.event(node, 'output', output=output)
                if 'final' in output:
                    final = output['final']
    if final is None:
        raise RuntimeError('No final assessment produced')
    elapsed = write_response(final, args.output / 'result.json', args.started_at)
    args.metrics.update(claim_to_response_seconds=elapsed, verdict=final.public_result()['verdict'],
                        confidence=final.public_result()['confidence'])
    trace.event('response_timing', 'output', claim_to_response_seconds=elapsed)
    failed = [a.paper_id for a in final.analyses if a.status == 'failed']
    args.metrics['failed_paper_analyses'] = len(failed)
    if failed or not final.sources:
        raise RuntimeError(f'Live verification incomplete: failed papers={failed}, sources={len(final.sources)}')
    trace.event('pipeline', 'passed', verdict=final.verdict, papers=len(papers), chunks=len(chunks), sources=len(final.sources))
    args.metrics['status'] = 'passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', default='coffee cardiovascular disease')
    parser.add_argument('--claim', default='Coffee consumption is associated with lower cardiovascular disease risk.')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-chunks-per-paper', type=int, default=3)
    parser.add_argument('--max-total-chunks', type=int, default=4)
    parser.add_argument('--max-chars-per-paper', type=int, default=8000)
    parser.add_argument('--langfuse', action='store_true', help='Upload stage and LLM traces to Langfuse using .env credentials')
    parser.add_argument('--output', type=Path, default=Path('outputs/logs') / datetime.now().strftime('live-%Y%m%d-%H%M%S'))
    args = parser.parse_args()
    args.started_at = time.perf_counter()
    args.metrics = dict(claim=args.claim, query=args.query, paper_limit=args.limit,
        workers=args.workers, top_k=args.top_k,
        max_chunks_per_paper=args.max_chunks_per_paper, max_total_chunks=args.max_total_chunks,
        max_chars_per_paper=args.max_chars_per_paper, papers_fetched=0, papers_with_evidence=0,
        chunks_indexed=0, chunks_retrieved=0, chunks_used_for_analysis=0,
        papers_analyzed=0, effective_worker_limit=0, status='failed')
    cloud = None
    handler = None
    if args.langfuse:
        from dotenv import load_dotenv
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
        load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=False)
        for name in ('LANGFUSE_SECRET_KEY', 'LANGFUSE_PUBLIC_KEY', 'LANGFUSE_BASE_URL'):
            if not os.getenv(name):
                parser.error(f'Missing {name} in environment or .env')
        def mask(data, **kwargs):
            return json.loads(Trace(Path('/dev/null')).redact(json.dumps(data, default=str)))
        cloud = Langfuse(mask=mask)
        if not cloud.auth_check():
            parser.error('Langfuse authentication failed')
        handler = CallbackHandler()
    args.output.mkdir(parents=True, exist_ok=False)
    trace = Trace(args.output / 'trace.jsonl', cloud=cloud)
    trace.callbacks = [handler] if handler else []
    with (args.output / 'pipeline.log').open('w', encoding='utf-8') as log:
        with redirect_stdout(Tee(sys.stdout, log, trace)), redirect_stderr(Tee(sys.stderr, log, trace)):
            print(f'Artifacts: {args.output.resolve()}', flush=True)
            try:
                with trace.stage('pipeline'):
                    if cloud:
                        url = cloud.get_trace_url()
                        trace.event('langfuse', 'connected', url=url)
                        print(f'Langfuse trace: {url}', flush=True)
                        (args.output / 'langfuse_trace.txt').write_text(url + '\n')
                    run(args, trace)
            except Exception as exc:
                args.metrics['error'] = trace.redact(f'{type(exc).__name__}: {exc}')
                print(f'FAILED: {type(exc).__name__}: {exc}', flush=True)
                return 1
            finally:
                args.metrics['elapsed_seconds'] = round(time.perf_counter() - args.started_at, 3)
                (args.output / 'metrics.json').write_text(trace.redact(json.dumps(args.metrics, indent=2)), encoding='utf-8')
                print('Run metrics: ' + json.dumps(args.metrics), flush=True)
                trace.event('run_metrics', 'info', **args.metrics)
                if cloud:
                    cloud.flush()
    return 0


if __name__ == '__main__':
    sys.exit(main())
