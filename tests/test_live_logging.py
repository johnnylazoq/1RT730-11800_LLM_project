"""Offline checks for the live runner's diagnostic safeguards."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('live_runner', Path(__file__).parents[1] / 'scripts/test_live_pipeline.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_trace_redacts_credentials_and_records_failed_stage(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('SEMANTIC_SCHOLAR_API_KEY', 'private-test-credential')
    trace = runner.Trace(tmp_path / 'trace.jsonl')
    with pytest.raises(ValueError):
        with trace.stage('search'):
            raise ValueError('request failed: private-test-credential')
    contents = trace.path.read_text()
    assert 'private-test-credential' not in contents
    events = [json.loads(line) for line in contents.splitlines()]
    assert [event['event'] for event in events] == ['start', 'error']
    assert events[-1]['message'] == 'request failed: [REDACTED]'
    assert events[-1]['seconds'] >= 0
    assert 'search: error' in capsys.readouterr().out


def test_llm_context_is_configurable(monkeypatch):
    from app.llm.client import get_llm
    monkeypatch.setenv('OLLAMA_NUM_CTX', '32768')
    assert get_llm().num_ctx == 32768


def test_parallel_analysis_preserves_trace_context(monkeypatch):
    from contextvars import ContextVar
    from app.graph import workflow
    from tests.test_pipeline import FakeLLM

    trace_parent = ContextVar('test_trace_parent', default=None)
    observed = []
    original = workflow.analyze_single_paper

    def analyze(*args, **kwargs):
        observed.append(trace_parent.get())
        return original(*args, **kwargs)

    monkeypatch.setattr(workflow, 'analyze_single_paper', analyze)
    chunks = json.loads((Path(__file__).parent / 'fixtures/retrieved_chunks.json').read_text())
    token = trace_parent.set('pipeline-root')
    try:
        workflow.verify_claim('Synthetic claim', chunks, llm=FakeLLM(), max_workers=2)
    finally:
        trace_parent.reset(token)
    assert observed == ['pipeline-root', 'pipeline-root']


def test_cloud_stage_and_masked_outputs(tmp_path, monkeypatch):
    from contextlib import contextmanager

    class Cloud:
        def __init__(self):
            self.events = []

        @contextmanager
        def start_as_current_observation(self, **kwargs):
            self.events.append(('start', kwargs['name']))
            try:
                yield
            finally:
                self.events.append(('end', kwargs['name']))

        def update_current_span(self, **kwargs):
            self.events.append(('output', kwargs['output']))

    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'secret-test-value')
    cloud = Cloud()
    trace = runner.Trace(tmp_path / 'trace.jsonl', cloud=cloud)
    with trace.stage('retrieval'):
        trace.event('retrieval', 'output', content='secret-test-value')
    assert cloud.events[0] == ('start', 'retrieval')
    assert cloud.events[1][1]['content'] == '[REDACTED]'
    assert cloud.events[-1] == ('end', 'retrieval')
