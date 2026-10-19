from langchain_core.messages import SystemMessage, HumanMessage
import json
from app.llm.context import check_context, ContextBudgetExceeded
from app.llm.client import get_llm
from app.llm.schemas import ScientificPaper, PaperAssessment, PaperAnalysis
from app.llm.prompts import PAPER_ANALYSIS_SYSTEM_PROMPT
from app.llm.citations import validate_analysis


def analyze_single_paper(claim: str, paper: ScientificPaper, *, llm=None, retry_feedback=None) -> PaperAnalysis:
    if not paper.chunks:
        return PaperAnalysis(paper_id=paper.paper_id, title=paper.title,
                             assessment=PaperAssessment(stance="insufficient",
                                                        explanation="No evidence chunks provided."))
    if any(chunk.paper_id != paper.paper_id for chunk in paper.chunks):
        raise ValueError("Chunk belongs to a different paper")
    client = llm if llm is not None else get_llm()
    model = client.with_structured_output(
        PaperAssessment, method="json_schema")
    evidence = {"paper_id": paper.paper_id, "title": paper.title,
                "chunks": [{"chunk_id": c.chunk_id, "text": c.text, "page_number": c.page_number,
                            "evidence_type": c.metadata.get("evidence_type")}
                           for c in paper.chunks]}
    messages = [
        SystemMessage(content=PAPER_ANALYSIS_SYSTEM_PROMPT),
        HumanMessage(content=f"Scientific claim:\n{claim}\n\nPaper evidence JSON:\n{json.dumps(evidence, ensure_ascii=False)}"),
    ]
    if retry_feedback:
        messages.append(HumanMessage(content=(
            "The previous response failed validation. Return a complete structured assessment "
            "with findings containing exact chunk IDs from the supplied paper. "
            f"Validation error type: {retry_feedback}")))
    omitted = 0
    while True:
        try:
            check_context(messages, PaperAssessment, client)
            break
        except ContextBudgetExceeded:
            if not evidence['chunks']:
                raise
            evidence['chunks'].pop()
            omitted += 1
            messages[1] = HumanMessage(content=f"Scientific claim:\n{claim}\n\nPaper evidence JSON:\n{json.dumps(evidence, ensure_ascii=False)}")
    if not evidence['chunks']:
        return PaperAnalysis(paper_id=paper.paper_id, title=paper.title,
            assessment=PaperAssessment(stance="insufficient", explanation="Evidence did not fit the request context budget.",
                limitations=["Context budget excluded all available chunks for this paper."]))
    assessment = model.invoke(messages)
    assessment = PaperAssessment.model_validate(assessment)
    if omitted:
        assessment.limitations.append(f"Context budget omitted {omitted} evidence chunks from this paper.")
    result = PaperAnalysis(paper_id=paper.paper_id, title=paper.title, assessment=assessment)
    sent_ids = {chunk['chunk_id'] for chunk in evidence['chunks']}
    validate_analysis(paper.model_copy(update={
        'chunks': [chunk for chunk in paper.chunks if chunk.chunk_id in sent_ids]}), result)
    return result
