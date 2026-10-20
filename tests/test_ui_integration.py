import importlib.util
from pathlib import Path
import sys

import pytest
from tests.test_pipeline import FakeLLM
from app.llm.schemas import RetrievedChunk

spec = importlib.util.spec_from_file_location('ui_pipeline', Path(__file__).parents[1] / 'ui/rumor2reality/pipeline.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


@pytest.fixture
def backend(monkeypatch):
    from app.rag import collect_papers, ingest, retriever
    chunks = [RetrievedChunk(paper_id='paper', chunk_id='paper:abstract', title='Test paper',
        text='Synthetic evidence.', score=0.7, metadata={'evidence_type': 'abstract'})]
    monkeypatch.setattr(collect_papers, 'fetch_papers', lambda *a, **kw: [])
    monkeypatch.setattr(collect_papers, 'papers_to_chunks', lambda papers: chunks)
    monkeypatch.setattr(ingest, 'ingest_chunks', lambda *a: None)
    calls = []
    class Retriever:
        def retrieve(self, claim, k):
            calls.append((claim, k))
            return chunks
    monkeypatch.setattr(retriever, 'create_retriever', lambda *a: Retriever())
    return chunks, calls


def test_ui_contract_real_graph_and_citations(backend):
    chunks, calls = backend
    events = list(pipeline.pipeline_events('A claim', 3, 'test-model', llm=FakeLLM()))
    assert [stage for stage, _ in events] == ['scrape', 'retrieve', 'generate', 'done']
    result = events[-1][1]
    assert calls == [('A claim', 3)]
    assert result['verdict']['label'] == 'Mixed evidence'
    assert result['sources'][0]['snippet'] == chunks[0].text
    assert result['sources'][0]['section'] == 'abstract'
    assert result['sources'][0]['rank'] == '1'
    assert result['evidence_summary'][0].endswith('[1]')
    assert any('abstracts' in note for note in result['limitations'])


def test_no_evidence_is_insufficient(backend):
    backend[0].clear()
    result = pipeline.run_pipeline('A claim', llm=FakeLLM())
    assert result['verdict']['label'] == 'Not enough evidence'
    assert result['sources'] == []


def test_failed_llm_is_an_error(backend):
    with pytest.raises(RuntimeError, match='Ollama'):
        pipeline.run_pipeline('A claim', llm=FakeLLM(fail_paper='paper'))


def test_existing_index_and_invalid_path(backend, tmp_path):
    with pytest.raises(ValueError, match='vector store directory'):
        pipeline.run_pipeline('A claim', evidence_source='Existing vector store', vector_store_path=str(tmp_path))
    (tmp_path / 'index.faiss').touch()
    (tmp_path / 'metadata.json').write_text('[]')
    result = pipeline.run_pipeline('A claim', evidence_source='Existing vector store',
                                   vector_store_path=str(tmp_path), llm=FakeLLM())
    assert result['metadata']['pipeline_mode'] == 'Existing vector store'


def test_retriever_preserves_nested_provenance():
    import numpy as np
    from app.rag.retriever import RAGRetriever
    class Store:
        def search(self, query, k):
            return [(0, 0.5, {'paper_id': 'p', 'chunk_id': 'c', 'text': 'Evidence',
                             'metadata': {'evidence_type': 'abstract', 'synthetic': True}})]
    class Embeddings:
        def embed_single(self, query):
            return np.zeros(3)
    chunk = RAGRetriever(Store(), Embeddings()).retrieve('claim')[0]
    assert chunk.metadata == {'evidence_type': 'abstract', 'synthetic': True}


def test_discovery_index_retrieval_graph_round_trip(monkeypatch):
    import numpy as np
    from app.rag import collect_papers, ingest, retriever
    class Embeddings:
        embedding_dim = 3
        def __init__(self, *args):
            pass
        def embed(self, texts):
            return np.array([[1, 0, 0] for text in texts], dtype=np.float32)
        def embed_single(self, text):
            return self.embed([text])[0]
    monkeypatch.setattr(ingest, 'EmbeddingModel', Embeddings)
    monkeypatch.setattr(retriever, 'EmbeddingModel', Embeddings)
    monkeypatch.setattr(collect_papers, 'fetch_papers', lambda *a, **kw: [
        {'paperId': 'p', 'title': 'Round trip', 'abstract': 'Synthetic abstract evidence.',
         'url': 'https://example.org/p', 'year': 2024, 'authors': [{'name': 'Author'}]}])
    result = pipeline.run_pipeline('A claim', llm=FakeLLM())
    source = result['assessment']['sources'][0]
    assert source['paper_id'] == 'p'
    assert source['authors'] == ['Author']
    assert source['metadata']['evidence_type'] == 'abstract'
    assert result['sources'][0]['relevance'] == 1.0
