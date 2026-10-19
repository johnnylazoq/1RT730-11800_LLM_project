import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from langchain_core.messages import HumanMessage
from app.llm.context import check_context, ContextBudgetExceeded
from app.llm.schemas import EvidenceSummaryItem, AggregationAssessment, PaperAssessment, PaperAnalysis
from app.graph.nodes.aggregate_evidence import aggregate_evidence
from app.graph.nodes.analyze_paper import analyze_single_paper
from app.graph.nodes.group_papers import group_chunks
from app.graph.workflow import verify_claim
from tests.test_pipeline import FakeLLM, chunks


@pytest.mark.parametrize('value', [{}, {'citation_ids': []}])
def test_citations_are_required_and_nonempty(value):
    with pytest.raises(ValidationError):
        EvidenceSummaryItem(statement='A finding.', **value)
    schema = EvidenceSummaryItem.model_json_schema()
    assert 'citation_ids' in schema['required']
    assert schema['properties']['citation_ids']['minItems'] == 1


def test_only_final_cited_papers_are_public(chunks):
    result = verify_claim('claim', chunks, llm=FakeLLM())
    for finding in result.key_findings:
        finding.citation_ids = ['2']
    assert len(result.sources) == 2  # Full diagnostic registry stays intact.
    assert result.public_result()['sources'] == [
        {'title': result.sources[1].title, 'url': result.sources[1].url}]


def test_global_and_per_paper_chunk_limits(chunks):
    records = [chunk.model_copy(update={'chunk_id': f'{chunk.paper_id}:{i}'})
               for chunk in chunks[:2] for i in range(10)]
    # Use two distinct papers irrespective of fixture ordering.
    records += [chunks[-1].model_copy(update={'chunk_id': f'last:{i}'}) for i in range(10)]
    papers, notes = group_chunks(records, max_chunks_per_paper=2, max_total_chunks=3)
    assert sum(len(p.chunks) for p in papers) == 3
    assert all(len(p.chunks) <= 2 for p in papers)
    assert notes


def test_complete_request_budget_includes_schema_and_reserve():
    llm = SimpleNamespace(num_ctx=4096)
    with pytest.raises(ContextBudgetExceeded, match='reserved for response'):
        check_context([HumanMessage(content='x' * 4096)], AggregationAssessment, llm)


def test_large_excerpts_are_not_repeated_in_aggregation(chunks):
    large = chunks[0].model_copy(update={'text': 'Full scientific excerpt. ' * 3000})
    papers, _ = group_chunks([large], max_chars_per_paper=100000)
    analyses = [PaperAnalysis(paper_id=large.paper_id, title=large.title,
        assessment=PaperAssessment(stance='mixed', explanation='An analysis.',
            findings=[{'statement': 'A complete concise finding.', 'chunk_id': large.chunk_id}]))]

    class Inspect(FakeLLM):
        def with_structured_output(self, schema, **kwargs):
            citation_schema = schema['$defs']['EvidenceSummaryItem']
            assert citation_schema['properties']['citation_ids']['items']['enum'] == ['1']
            assert 'citation_ids' in citation_schema['required']
            delegate = super().with_structured_output(schema, **kwargs)
            def invoke(messages):
                assert 'Full scientific excerpt' not in messages[1].content
                payload = json.loads(messages[1].content)
                assert payload['analyses'][0]['assessment']['findings'][0]['statement'] == 'A complete concise finding.'
                assert 'excerpt' not in payload['sources'][0]
                return delegate.invoke(messages)
            return SimpleNamespace(invoke=invoke)

    result = aggregate_evidence('claim', papers, analyses, llm=Inspect())
    assert result.sources[0].excerpt == large.text


def test_paper_context_omits_oversized_chunk_without_truncating_text(chunks):
    paper = group_chunks(chunks[:1])[0][0]
    paper.chunks.append(paper.chunks[0].model_copy(update={'chunk_id': 'large', 'text': 'x' * 30000}))
    result = analyze_single_paper('claim', paper, llm=FakeLLM())
    assert any('Context budget omitted 1' in note for note in result.assessment.limitations)
    assert len(paper.chunks) == 2
    assert result.assessment.findings[0].chunk_id == paper.chunks[0].chunk_id


def test_cannot_cite_a_chunk_removed_by_request_budget(chunks):
    paper = group_chunks(chunks[:1])[0][0]
    paper.chunks.append(paper.chunks[0].model_copy(update={'chunk_id': 'removed', 'text': 'x' * 30000}))

    class WrongCitation(FakeLLM):
        def with_structured_output(self, schema, **kwargs):
            return SimpleNamespace(invoke=lambda messages: {
                'stance': 'supports', 'explanation': 'A finding.',
                'findings': [{'statement': 'A finding.', 'chunk_id': 'removed'}]})

    with pytest.raises(ValueError, match='Unknown citation'):
        analyze_single_paper('claim', paper, llm=WrongCitation())


def test_aggregate_budget_preserves_all_paper_stances(chunks):
    class InspectingLLM(FakeLLM):
        num_ctx = 12000
        observed = None

        def with_structured_output(self, schema, **kwargs):
            delegate = super().with_structured_output(schema, **kwargs)
            def invoke(messages):
                self.observed = json.loads(messages[1].content)
                return delegate.invoke(messages)
            return SimpleNamespace(invoke=invoke)

    papers, _ = group_chunks(chunks)
    analyses = [analyze_single_paper('claim', paper, llm=FakeLLM()) for paper in papers]
    # Simulate a legacy diagnostic record larger than today's output schema.
    for analysis in analyses:
        analysis.assessment = analysis.assessment.model_copy(update={
            'quality_considerations': ['Detailed quality note. ' * 1000]})
    llm = InspectingLLM()
    aggregate_evidence('claim', papers, analyses, llm=llm)
    assert {p['paper_id'] for p in llm.observed['analyses']} == {'synthetic-a', 'synthetic-b'}
    assert all(p['assessment']['stance'] == 'mixed' for p in llm.observed['analyses'])
    assert all('quality_considerations' not in p['assessment'] for p in llm.observed['analyses'])
    assert any('Context budget' in note for note in llm.observed['limitations'])
