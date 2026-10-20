"""Backend regressions for document verification and page/paragraph provenance."""
from io import BytesIO
import json
from urllib.error import HTTPError

import numpy as np
import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from app.rag import collect_papers, ingest
from app.rag.chunk_documents import chunk_pages
from app.rag.extract_documents import DocumentPage, extract_pages_from_pdf
from app.rag.retriever import RAGRetriever
from app.rag.vector_store import VectorStore
from app.graph.nodes.group_papers import group_chunks
from app.llm.citations import source_registry
from app.llm.schemas import PaperAnalysis, PaperAssessment, EvidenceFinding


def make_pdf():
    """Real three-page PDF with text and a blank middle page, no external files."""
    writer = PdfWriter()
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                             NameObject('/Subtype'): NameObject('/Type1'),
                             NameObject('/BaseFont'): NameObject('/Helvetica')})
    for lines in [('First paragraph.', 'Second paragraph.'), (), ('Third page evidence.',)]:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject('/Resources')] = DictionaryObject({
            NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        commands = ['BT /F1 12 Tf']
        for i, line in enumerate(lines):
            commands.append(f'1 0 0 1 72 {720-i*60} Tm ({line}) Tj')
        commands.append('ET')
        stream.set_data('\n'.join(commands).encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class Response(BytesIO):
    headers = {'Content-Type': 'application/pdf'}

    def geturl(self):
        return 'https://example.org/paper.pdf'


@pytest.fixture
def paper():
    return {'paperId': 'p1', 'title': 'Study', 'authors': [{'name': 'Author'}],
            'abstract': 'Abstract only.', 'year': 2024, 'externalIds': {'DOI': '10/example'},
            'openAccessPdf': {'url': 'https://example.org/paper.pdf'}, 'isOpenAccess': True}


def test_real_pdf_extraction_preserves_blank_page_and_paragraphs():
    pages = extract_pages_from_pdf(make_pdf())
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert pages[1].text == ''
    chunks = chunk_pages(pages, 'p1')
    assert [(c.page_number, c.paragraph_number) for c in chunks] == [(1, 1), (1, 2), (3, 1)]
    assert [c.text for c in chunks] == ['First paragraph.', 'Second paragraph.', 'Third page evidence.']
    assert chunks[-1].chunk_id == 'p1:page_3:paragraph_1'
    assert chunks == chunk_pages(pages, 'p1')


def test_page_strategy_and_long_paragraphs():
    pages = [DocumentPage(1, 'x'*3000 + '\n\nSecond.'), DocumentPage(2, '')]
    paragraphs = chunk_pages(pages, 'p1')
    assert len(paragraphs) == 2 and len(paragraphs[0].text) == 3000
    whole_page = chunk_pages(pages, 'p1', 'page')
    assert len(whole_page) == 1
    assert whole_page[0].text == pages[0].text
    assert whole_page[0].paragraph_number is None
    with pytest.raises(ValueError):
        chunk_pages(pages, 'p1', 'fixed')


def test_verified_pdf_without_abstract(monkeypatch, paper):
    paper['abstract'] = None
    requests = []
    def download(request, timeout):
        requests.append(request)
        return Response(make_pdf())
    monkeypatch.setenv('SEMANTIC_SCHOLAR_API_KEY', 'test-key')
    monkeypatch.setattr(collect_papers, 'urlopen', download)
    chunks, report = collect_papers.resolve_paper(paper)
    assert report['full_text_status'] == 'pdf_text_available'
    assert report['document_format'] == 'pdf'
    assert report['empty_pages'] == [2]
    assert len(chunks) == 3
    assert chunks[-1].page_number == 3
    assert chunks[0].doi == '10/example'
    assert not any(k.lower() == 'x-api-key' for k in requests[0].headers)


@pytest.mark.parametrize('kind,status', [
    ('missing', 'no_pdf_url'), ('html', 'non_pdf_response'),
    ('blocked', 'download_failed'), ('corrupt', 'pdf_extraction_failed'),
    ('blank', 'pdf_without_extractable_text'), ('oversize', 'download_too_large'),
])
def test_fallback_is_explicit(monkeypatch, paper, kind, status):
    def download(request, timeout):
        if kind == 'blocked':
            raise HTTPError(request.full_url, 403, 'Forbidden', {}, None)
        if kind == 'blank':
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            data = BytesIO()
            writer.write(data)
            return Response(data.getvalue())
        return Response({'html': b'<html>Login</html>', 'corrupt': b'%PDF-invalid',
                         'oversize': b'x'*21}.get(kind, b''))
    if kind == 'missing':
        paper['openAccessPdf'] = None
    if kind == 'oversize':
        monkeypatch.setattr(collect_papers, 'MAX_PDF_BYTES', 20)
    monkeypatch.setattr(collect_papers, 'urlopen', download)
    chunks, report = collect_papers.resolve_paper(paper)
    assert report['full_text_status'] == status
    assert chunks[0].metadata['evidence_type'] == 'abstract'
    assert chunks[0].metadata['full_text_status'] == status
    assert chunks[0].page_number is None
    assert collect_papers.resolve_paper(paper, abstract_fallback=False)[0] == []


def test_search_pagination_and_retry(monkeypatch):
    payloads = [{'data': [{'paperId': 'p1'}], 'next': 1}, {'data': [{'paperId': 'p2'}]}]
    requests = []
    def download(request, timeout):
        requests.append(request)
        if len(requests) == 1:
            raise HTTPError(request.full_url, 429, 'Rate limited', {'Retry-After': '0'}, None)
        return Response(json.dumps(payloads.pop(0)).encode())
    monkeypatch.setattr(collect_papers, 'urlopen', download)
    assert [p['paperId'] for p in collect_papers.fetch_papers('topic', 2)] == ['p1', 'p2']
    assert 'offset=1' in requests[-1].full_url
    assert 'openAccessPdf' in requests[-1].full_url


class TinyEmbedding:
    """Deterministic embedding for testing storage/provenance, not relevance."""
    embedding_dim = 3
    def __init__(self, *args):
        pass
    def embed(self, texts):
        return np.array([[len(text), 1, 0] for text in texts], dtype=np.float32)
    def embed_single(self, text):
        return self.embed([text])[0]


def test_api_ingestion_store_reload_retrieval_and_citations(monkeypatch, tmp_path, paper):
    monkeypatch.setattr(ingest, 'fetch_papers', lambda *a, **kw: [paper])
    monkeypatch.setattr(collect_papers, 'urlopen', lambda *a, **kw: Response(make_pdf()))
    monkeypatch.setattr(ingest, 'EmbeddingModel', TinyEmbedding)
    ingest.ingest_papers_from_api('topic', str(tmp_path))
    report = json.loads((tmp_path / 'ingestion_report.json').read_text())
    assert report[0]['chunk_count'] == 3
    store = VectorStore(embedding_dim=3)
    store.load(str(tmp_path))
    records = RAGRetriever(store, TinyEmbedding()).retrieve('Third page evidence.', k=3)
    assert records[0].page_number == 3
    assert records[0].metadata['evidence_type'] == 'pdf_chunk'
    assert records[0].metadata['paragraph_number'] == 1
    assert 'start_char' not in records[0].metadata
    papers, _ = group_chunks(records)
    analysis = PaperAnalysis(paper_id='p1', title='Study', assessment=PaperAssessment(
        stance='supports', explanation='Example', findings=[EvidenceFinding(
            statement='Example', chunk_id=records[0].chunk_id)]))
    sources = source_registry(papers, [analysis])
    assert sources[0].page_number == 3
    assert sources[0].excerpt == 'Third page evidence.'


def test_local_ingestion_keeps_pages(monkeypatch, tmp_path):
    folder = tmp_path / 'pdfs'
    folder.mkdir()
    (folder / 'paper.pdf').write_bytes(make_pdf())
    monkeypatch.setattr(ingest, 'EmbeddingModel', TinyEmbedding)
    store = ingest.ingest_pdfs(str(folder), str(tmp_path / 'index'))
    assert {m['page_number'] for m in store.id_to_metadata.values()} == {1, 3}
    assert all('start_char' not in m['metadata'] for m in store.id_to_metadata.values())


def test_abstract_metadata_survives_store(monkeypatch, tmp_path, paper):
    paper['openAccessPdf'] = None
    chunks, _ = collect_papers.resolve_paper(paper)
    monkeypatch.setattr(ingest, 'EmbeddingModel', TinyEmbedding)
    store = ingest.ingest_chunks(chunks, str(tmp_path))
    retrieved = RAGRetriever(store, TinyEmbedding()).retrieve('Abstract only.')
    assert retrieved[0].metadata['evidence_type'] == 'abstract'
    assert retrieved[0].metadata['full_text_status'] == 'no_pdf_url'


def test_rag_workflow_preserves_page_and_abstract_warning(monkeypatch, tmp_path, paper):
    from app.graph.workflow import verify_claim
    from tests.test_pipeline import FakeLLM
    monkeypatch.setattr(ingest, 'EmbeddingModel', TinyEmbedding)
    monkeypatch.setattr(collect_papers, 'urlopen', lambda *a, **kw: Response(make_pdf()))
    pdf_chunks, _ = collect_papers.resolve_paper(paper)
    abstract_paper = {**paper, 'paperId': 'p2', 'openAccessPdf': None}
    abstracts, _ = collect_papers.resolve_paper(abstract_paper)
    store = ingest.ingest_chunks(pdf_chunks + abstracts, str(tmp_path))
    result = verify_claim('Third page evidence.', [],
        retriever=RAGRetriever(store, TinyEmbedding()), llm=FakeLLM(), retries=0)
    assert any(source.page_number == 3 for source in result.sources)
    assert any('abstracts rather than full papers' in note for note in result.limitations)


def test_empty_fulltext_ingestion_still_reports_reason(monkeypatch, tmp_path, paper):
    paper['openAccessPdf'] = None
    monkeypatch.setattr(ingest, 'fetch_papers', lambda *a, **kw: [paper])
    with pytest.raises(ValueError, match='No chunks to ingest'):
        ingest.ingest_papers_from_api('topic', str(tmp_path), abstract_fallback=False)
    report = json.loads((tmp_path / 'ingestion_report.json').read_text())
    assert report[0]['full_text_status'] == 'no_pdf_url'
    assert report[0]['evidence_type'] == 'none'
    assert not (tmp_path / 'index.faiss').exists()


def test_arxiv_identifier_when_semantic_scholar_has_no_pdf_link(monkeypatch, paper):
    paper['openAccessPdf'] = None
    paper['externalIds']['ArXiv'] = '2005.11401'
    def download(request, timeout):
        assert request.full_url == 'https://arxiv.org/pdf/2005.11401'
        return Response(make_pdf())
    monkeypatch.setattr(collect_papers, 'urlopen', download)
    chunks, report = collect_papers.resolve_paper(paper)
    assert chunks and report['full_text_status'] == 'pdf_text_available'
    assert report['pdf_url_source'] == 'externalIds.ArXiv'
