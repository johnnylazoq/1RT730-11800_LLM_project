from langchain_core.messages import SystemMessage, HumanMessage
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
    model = (llm if llm is not None else get_llm()).with_structured_output(
        PaperAssessment, method="json_schema")
    messages = [
        SystemMessage(content=PAPER_ANALYSIS_SYSTEM_PROMPT),
        HumanMessage(content=f"Scientific claim:\n{claim}\n\nPaper evidence JSON:\n{paper.model_dump_json()}"),
    ]
    if retry_feedback:
        messages.append(HumanMessage(content=(
            "The previous response failed validation. Return a complete structured assessment "
            "with findings containing exact chunk IDs from the supplied paper. "
            f"Validation error type: {retry_feedback}")))
    assessment = model.invoke(messages)
    assessment = PaperAssessment.model_validate(assessment)
    result = PaperAnalysis(paper_id=paper.paper_id, title=paper.title, assessment=assessment)
    validate_analysis(paper, result)
    return result
