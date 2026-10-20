"""Check bounded paper responses and embedding reuse without model downloads."""
import pytest
from pydantic import ValidationError
from app.llm.schemas import PaperAssessment
from app.rag import ingest, retriever
from tests.test_pipeline import chunks
from tests.test_fulltext_pipeline import TinyEmbedding


def test_embedding_instance_reused_for_ingestion_and_retrieval(chunks, tmp_path, monkeypatch):
    def unexpected_load(*args, **kwargs):
        raise AssertionError('A second embedding model was loaded')

    monkeypatch.setattr(ingest, 'EmbeddingModel', unexpected_load)
    monkeypatch.setattr(retriever, 'EmbeddingModel', unexpected_load)
    model = TinyEmbedding()
    records = [chunks[0].model_copy(update={'chunk_id': f'chunk-{i}'}) for i in range(6)]
    ingest.ingest_chunks(records, str(tmp_path), embedding_model=model)
    loaded = retriever.create_retriever(str(tmp_path), embedding_model=model)
    assert loaded.embedding_model is model
    found = loaded.retrieve('evidence')
    assert len(found) == 4
    assert all(chunk.page_number == chunks[0].page_number for chunk in found)


@pytest.mark.parametrize('field,value', [
    ('explanation', 'x' * 241),
    ('findings', [{'statement': 'A finding', 'chunk_id': '1'}] * 4),
    ('limitations', ['one', 'two', 'three']),
    ('quality_considerations', ['one', 'two']),
])
def test_paper_analysis_rejects_excessive_detail(field, value):
    payload = {'stance': 'insufficient', 'explanation': 'Evidence is limited.'}
    payload[field] = value
    with pytest.raises(ValidationError):
        PaperAssessment.model_validate(payload)
