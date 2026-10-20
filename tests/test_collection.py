import io
from urllib.error import HTTPError

import pytest
from app.rag import collect_papers


def rate_limit(retry_after='0'):
    return HTTPError('https://example.org', 429, 'Too many requests', {'Retry-After': retry_after}, None)


def test_rate_limit_recovers(monkeypatch):
    calls = []
    sleeps = []
    def request(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise rate_limit('2')
        return io.BytesIO(b'{"data": [{"paperId": "p"}]}')
    monkeypatch.setattr(collect_papers, 'urlopen', request)
    monkeypatch.setattr(collect_papers.time, 'sleep', sleeps.append)
    assert collect_papers.fetch_papers('claim') == [{'paperId': 'p'}]
    assert sleeps == [2]


def test_rate_limit_stops_and_explains(monkeypatch):
    calls = []
    def request(*args, **kwargs):
        calls.append(1)
        raise rate_limit()
    monkeypatch.setattr(collect_papers, 'urlopen', request)
    monkeypatch.setattr(collect_papers.time, 'sleep', lambda delay: None)
    with pytest.raises(RuntimeError, match='claim has not been assessed'):
        collect_papers.fetch_papers('claim')
    assert len(calls) == 3


def test_long_retry_after_is_not_ignored(monkeypatch):
    def request(*args, **kwargs):
        raise rate_limit('120')
    monkeypatch.setattr(collect_papers, 'urlopen', request)
    monkeypatch.setattr(collect_papers.time, 'sleep', lambda delay: pytest.fail('Must not retry early'))
    with pytest.raises(RuntimeError, match='rate-limiting'):
        collect_papers.fetch_papers('claim')
