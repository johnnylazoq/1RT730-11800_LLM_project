import json
from langchain_core.messages import SystemMessage, HumanMessage
from app.llm.client import get_llm
from app.llm.context import check_context, ContextBudgetExceeded
from app.llm.citations import source_registry
from app.llm.prompts import AGGREGATION_SYSTEM_PROMPT
from app.llm.schemas import AggregationAssessment, FinalAssessment


def aggregate_evidence(claim, papers, analyses, *, limitations=(), evidence_cutoff=None, llm=None, retry_feedback=None):
    sources = source_registry(papers, analyses)
    notes = list(limitations)
    if any(c.metadata.get("synthetic") for p in papers for c in p.chunks):
        notes.append("SYNTHETIC TEST EVIDENCE: this output is not a scientific assessment of real studies.")
    if evidence_cutoff is None:
        notes.append("Evidence cutoff is unknown; retrieved evidence may not be comprehensive.")
    if any(c.metadata.get("evidence_type") == "abstract" for p in papers for c in p.chunks):
        notes.append("Some or all evidence consists of abstracts rather than full papers.")
    failures = [a.paper_id for a in analyses if a.status == "failed"]
    if failures:
        notes.append("Paper analysis failed for: " + ", ".join(failures))
    if not sources:
        return FinalAssessment(claim=claim, verdict="insufficient", confidence="low",
            evidence_summary="No validated scientific findings were available for this claim. The available evidence cannot support or contradict it.",
            key_findings=[], limitations=[
                "No usable cited evidence was available for synthesis.",
                "Paper analysis failed; no findings could be validated." if failures else
                "The retrieved evidence may not cover the research relevant to this claim."],
            sources=[], analyses=analyses, evidence_cutoff=evidence_cutoff)
    client = llm if llm is not None else get_llm()
    schema = AggregationAssessment.model_json_schema()
    schema["$defs"]["EvidenceSummaryItem"]["properties"]["citation_ids"]["items"]["enum"] = [s.citation_id for s in sources]
    citation_map = {(s.paper_id, s.chunk_id): s.citation_id for s in sources}
    summaries = []
    for analysis in analyses:
        summary = analysis.model_dump(exclude={"title", "error"})
        summary["assessment"].pop("explanation", None)
        summary["assessment"]["findings"] = [
            {"statement": finding.statement,
             "citation_ids": [citation_map[(analysis.paper_id, finding.chunk_id)]]}
            for finding in analysis.assessment.findings
            if (analysis.paper_id, finding.chunk_id) in citation_map]
        summaries.append(summary)
    payload = {"claim": claim, "analyses": summaries,
               "sources": [s.model_dump(include={"citation_id", "paper_id"}) for s in sources],
               "allowed_citation_ids": [s.citation_id for s in sources],
               "limitations": notes, "evidence_cutoff": evidence_cutoff,
               "unique_papers_analyzed": len(analyses)}
    messages = [
        SystemMessage(content=AGGREGATION_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(payload)),
    ]
    if retry_feedback:
        messages.append(HumanMessage(content=f"Previous response failed validation: {retry_feedback}. Return a corrected complete response. Use only allowed_citation_ids from the evidence payload."))
    # Keep every paper's stance and at least one complete finding. Omit optional
    # detail progressively if the complete prompt/schema exceeds the input budget.
    for reduction in range(4):
        retained_ids = {citation for summary in summaries
                        for finding in summary['assessment']['findings']
                        for citation in finding['citation_ids']}
        payload['sources'] = [s.model_dump(include={'citation_id', 'paper_id'})
                              for s in sources if s.citation_id in retained_ids]
        payload['allowed_citation_ids'] = [s['citation_id'] for s in payload['sources']]
        schema['$defs']['EvidenceSummaryItem']['properties']['citation_ids']['items']['enum'] = payload['allowed_citation_ids']
        messages[1] = HumanMessage(content=json.dumps(payload, ensure_ascii=False))
        try:
            check_context(messages, schema, client)
            break
        except ContextBudgetExceeded:
            if reduction == 0:
                for summary in summaries:
                    summary['assessment'].pop('quality_considerations', None)
            elif reduction == 1:
                for summary in summaries:
                    summary['assessment']['findings'] = summary['assessment']['findings'][:2]
            elif reduction == 2:
                for summary in summaries:
                    summary['assessment']['findings'] = summary['assessment']['findings'][:1]
            else:
                raise
            note = "Context budget omitted some analysis details; coverage may be incomplete."
            if note not in notes:
                notes.append(note)
    model = client.with_structured_output(schema, method="json_schema")
    response = model.invoke(messages)
    assessment = AggregationAssessment.model_validate(response)
    known = set(payload["allowed_citation_ids"])
    for index, item in enumerate(assessment.key_findings):
        if not item.statement.strip():
            raise ValueError(f"key_findings[{index}].statement must not be blank")
        invalid = set(item.citation_ids) - known
        if invalid:
            raise ValueError(f"key_findings[{index}].citation_ids contains unknown citation {sorted(invalid)}; allowed IDs: {sorted(known)}")
    if len({item.statement.strip().casefold() for item in assessment.key_findings}) != len(assessment.key_findings):
        raise ValueError("Aggregate contains duplicate key findings")
    if len({item.strip().casefold() for item in assessment.limitations}) != len(assessment.limitations) or any(not item.strip() for item in assessment.limitations):
        raise ValueError("Aggregate contains empty or duplicate limitations")
    # Critical operational caveats must survive summarization.
    mandatory = []
    combined = " ".join(assessment.limitations).lower()
    if failures and "fail" not in combined:
        mandatory.append("Some paper analyses failed, reducing evidence coverage.")
    if any(c.metadata.get("evidence_type") == "abstract" for p in papers for c in p.chunks) and "abstract" not in combined:
        mandatory.append("Some evidence consists of abstracts rather than full papers.")
    if any(c.metadata.get("synthetic") for p in papers for c in p.chunks) and "synthetic test" not in combined:
        mandatory.append("SYNTHETIC TEST EVIDENCE: not a scientific assessment of real studies.")
    if mandatory:
        if len(assessment.limitations) < 3:
            assessment.limitations.append(" ".join(mandatory))
        else:
            assessment.limitations[-1] += " " + " ".join(mandatory)
    return FinalAssessment(claim=claim, **assessment.model_dump(),
        sources=sources, analyses=analyses, evidence_cutoff=evidence_cutoff)
