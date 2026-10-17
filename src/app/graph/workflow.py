from concurrent.futures import ThreadPoolExecutor
from langgraph.graph import StateGraph, START, END
from app.graph.state import GraphState
from app.graph.nodes.group_papers import group_chunks
from app.graph.nodes.analyze_paper import analyze_single_paper
from app.graph.nodes.aggregate_evidence import aggregate_evidence
from app.llm.schemas import RetrievedChunk, PaperAnalysis, PaperAssessment


def build_graph(*, llm=None, max_workers=3, retries=1, **group_options):
    """Bounded concurrent requests; transport timeout is configured in the LLM client."""
    if max_workers < 1 or retries < 0:
        raise ValueError("Invalid worker or retry count")

    def group(state):
        if not state["claim"].strip():
            raise ValueError("Claim must not be blank")
        records = [RetrievedChunk.model_validate(r) for r in state["retrieved_chunks"]]
        papers, notes = group_chunks(records, **group_options)
        return {"papers": papers, "limitations": notes}

    def analyze(state):
        def run(paper):
            error = None
            for attempt in range(retries + 1):
                try:
                    return analyze_single_paper(state["claim"], paper, llm=llm, retry_feedback=error)
                except Exception as exc:
                    error = type(exc).__name__
            return PaperAnalysis(paper_id=paper.paper_id, title=paper.title, status="failed",
                error=error, assessment=PaperAssessment(stance="insufficient",
                    explanation="Paper could not be analyzed due to a processing failure."))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return {"analyses": list(pool.map(run, state["papers"]))}

    def aggregate(state):
        # Fail closed: invalid aggregate citations never become a final response.
        for attempt in range(retries + 1):
            try:
                return {"final": aggregate_evidence(state["claim"], state["papers"],
                    state["analyses"], limitations=state["limitations"],
                    evidence_cutoff=state.get("evidence_cutoff"), llm=llm)}
            except Exception:
                if attempt == retries:
                    raise

    builder = StateGraph(GraphState)
    builder.add_node("group_papers", group)
    builder.add_node("analyze_papers", analyze)
    builder.add_node("aggregate_evidence", aggregate)
    builder.add_edge(START, "group_papers")
    builder.add_edge("group_papers", "analyze_papers")
    builder.add_edge("analyze_papers", "aggregate_evidence")
    builder.add_edge("aggregate_evidence", END)
    return builder.compile()


def verify_claim(claim, retrieved_chunks, *, evidence_cutoff=None, **options):
    return build_graph(**options).invoke({"claim": claim, "retrieved_chunks": retrieved_chunks,
                                         "evidence_cutoff": evidence_cutoff})["final"]
