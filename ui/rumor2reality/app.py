from __future__ import annotations

import html
import json
import re
from typing import Any

import gradio as gr

from pipeline import run_pipeline


DEFAULT_CLAIM = "A viral post says that most online rumors are created by bots."
READY_ANSWER = """
### Ready to check a rumor

Run the pipeline to see a concise verdict, selected evidence, and timing data.
"""
READY_TIMING = {
    "status": "ready",
    "next_action": "enter a claim and run the check",
}


def _empty_sources_html() -> str:
    return """
    <div class="r2r-empty-evidence">
      <strong>No evidence retrieved yet.</strong>
      <span>Run the check to see the papers selected by the retrieval step.</span>
    </div>
    """


def _sources_cards_html(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return _empty_sources_html()

    cards = []
    for source in sources:
        rank = html.escape(str(source["rank"]))
        title = html.escape(str(source["title"]))
        year = html.escape(str(source["year"]))
        relevance = int(float(source["relevance"]) * 100)
        reason = html.escape(str(source["why_selected"]))
        url = html.escape(str(source["url"]))
        stance = html.escape(str(source.get("stance", "Background")))
        study_type = html.escape(str(source.get("study_type", "Paper")))
        page = html.escape(str(source.get("page", "?")))
        section = html.escape(str(source.get("section", "Unknown section")))
        snippet = html.escape(str(source.get("snippet", "")))
        cards.append(
            f"""
            <article class="r2r-source-card">
              <div class="r2r-source-topline">
                <span class="r2r-source-rank">#{rank}</span>
                <span class="r2r-source-score">{relevance}% relevant</span>
              </div>
              <h3>{title} <span>({year})</span></h3>
              <div class="r2r-source-meta">
                <span>{stance}</span>
                <span>{study_type}</span>
                <span>p. {page}</span>
              </div>
              <p>{reason}</p>
              <blockquote>{snippet}</blockquote>
              <a href="{url}" target="_blank" rel="noreferrer">Open source - {section}</a>
            </article>
            """
        )
    return f"""<div class="r2r-sources-grid">{''.join(cards)}</div>"""


def _progress_html(active_step: str) -> str:
    steps = [
        ("scrape", "Scrape / load papers"),
        ("retrieve", "RAG retrieval"),
        ("generate", "LLM answer"),
        ("done", "Final response"),
    ]

    cards = []
    reached_active = False
    for key, label in steps:
        if active_step == "ready":
            state = "waiting"
            state_text = "Waiting"
        elif active_step == "done":
            state = "done"
            state_text = "Done"
        elif key == active_step:
            state = "active"
            state_text = "Running"
            reached_active = True
        elif reached_active:
            state = "waiting"
            state_text = "Waiting"
        else:
            state = "done"
            state_text = "Done"
        cards.append(
            f"""
            <div class="r2r-step r2r-step-{state}">
              <span class="r2r-step-dot"></span>
              <div>
                <strong>{label}</strong>
                <small>{state_text}</small>
              </div>
            </div>
            """
        )
    return f"""<div class="r2r-progress">{''.join(cards)}</div>"""


def _debug_json(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, ensure_ascii=True)


def _inline_markdown(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def _answer_html(
    markdown_text: str,
    tone: str = "ready",
    verdict: dict[str, str] | None = None,
    confidence: dict[str, str] | None = None,
    evidence_summary: list[str] | None = None,
    limitations: list[str] | None = None,
) -> str:
    blocks: list[str] = []
    for raw_block in markdown_text.strip().split("\n\n"):
        block = raw_block.strip()
        if not block:
            continue
        if block.startswith("### "):
            blocks.append(f"<h3>{_inline_markdown(block[4:])}</h3>")
        else:
            lines = [_inline_markdown(line) for line in block.splitlines()]
            blocks.append(f"<p>{'<br>'.join(lines)}</p>")

    verdict_html = ""
    if verdict:
        label = html.escape(verdict.get("label", "Not enough evidence"))
        rationale = html.escape(verdict.get("rationale", ""))
        verdict_tone = html.escape(verdict.get("tone", "caution"))
        verdict_html = f"""
        <div class="r2r-verdict r2r-verdict-{verdict_tone}">
          <span>{label}</span>
          <small>{rationale}</small>
        </div>
        """

    confidence_html = ""
    if confidence:
        label = html.escape(confidence.get("label", "Low"))
        rationale = html.escape(confidence.get("rationale", ""))
        confidence_html = f"""
        <div class="r2r-confidence">
          <strong>Confidence: {label}</strong>
          <span>{rationale}</span>
        </div>
        """

    summary_html = ""
    if evidence_summary:
        items = "".join(f"<li>{html.escape(item)}</li>" for item in evidence_summary)
        summary_html = f"""
        <div class="r2r-output-list">
          <strong>Evidence summary</strong>
          <ul>{items}</ul>
        </div>
        """

    limitations_html = ""
    if limitations:
        items = "".join(f"<li>{html.escape(item)}</li>" for item in limitations)
        limitations_html = f"""
        <div class="r2r-output-list r2r-limitations">
          <strong>Limitations</strong>
          <ul>{items}</ul>
        </div>
        """

    content = "".join(blocks)
    return (
        f"""<div class="r2r-answer-card r2r-answer-{tone}">"""
        f"{verdict_html}{confidence_html}{summary_html}{content}{limitations_html}</div>"
    )


def analyse_claim(claim: str, top_k: int, model_name: str):
    claim = (claim or "").strip()
    if not claim:
        yield (
            _progress_html("ready"),
            _answer_html("### Add a rumor or claim\n\nPaste a challenge prompt, then run the check.", "warning"),
            _empty_sources_html(),
            READY_TIMING,
            "",
        )
        return

    yield (
        _progress_html("scrape"),
        _answer_html("### Checking evidence\n\nLoading or scraping candidate papers...", "working"),
        _empty_sources_html(),
        {"status": "loading papers"},
        "",
    )

    yield (
        _progress_html("retrieve"),
        _answer_html("### Checking evidence\n\nRetrieving the most relevant papers for this claim...", "working"),
        _empty_sources_html(),
        {"status": "retrieving evidence"},
        "",
    )

    yield (
        _progress_html("generate"),
        _answer_html("### Checking evidence\n\nGenerating a grounded response from the retrieved evidence...", "working"),
        _empty_sources_html(),
        {"status": "generating answer"},
        "",
    )

    result = run_pipeline(claim=claim, top_k=int(top_k), model_name=model_name)
    yield (
        _progress_html("done"),
        _answer_html(
            result["answer"],
            "done",
            verdict=result["verdict"],
            confidence=result["confidence"],
            evidence_summary=result["evidence_summary"],
            limitations=result["limitations"],
        ),
        _sources_cards_html(result["sources"]),
        {"timings": result["timings"], "metadata": result["metadata"]},
        _debug_json(result),
    )


def clear_outputs():
    return _progress_html("ready"), _answer_html(READY_ANSWER), _empty_sources_html(), READY_TIMING, ""


CSS = """
.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(249, 115, 22, 0.14), transparent 28rem),
        linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%) !important;
    color: #172033;
}
.r2r-shell {
    max-width: 1160px;
    margin: 0 auto;
    padding: 2rem 0 2.5rem;
}
.r2r-title h1 {
    margin: 0 0 0.25rem;
    color: #172033;
    font-size: clamp(2rem, 4.2vw, 3.6rem);
    line-height: 1.08;
    letter-spacing: 0;
}
.r2r-title p {
    max-width: 760px;
    margin: 0.4rem 0 1rem;
    color: #475569;
    font-size: 1rem;
}
.r2r-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.8rem 0 1.2rem;
}
.r2r-badge {
    border: 1px solid #cbd5e1;
    background: rgba(255, 255, 255, 0.72);
    border-radius: 999px;
    color: #334155;
    font-size: 0.84rem;
    padding: 0.35rem 0.7rem;
}
.r2r-panel {
    border: 1px solid #d8dee8;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.8);
    box-shadow: 0 16px 42px rgba(15, 23, 42, 0.08);
}
.r2r-progress {
    display: grid;
    gap: 0.65rem;
}
.r2r-step {
    align-items: center;
    border: 1px solid #d8dee8;
    border-radius: 8px;
    display: grid;
    gap: 0.75rem;
    grid-template-columns: auto 1fr;
    min-height: 62px;
    padding: 0.8rem;
    background: #ffffff;
}
.r2r-step strong {
    display: block;
    color: #172033;
    font-size: 0.92rem;
}
.r2r-step small {
    color: #64748b;
}
.r2r-step-dot {
    width: 0.8rem;
    height: 0.8rem;
    border-radius: 50%;
    background: #cbd5e1;
}
.r2r-step-done .r2r-step-dot {
    background: #16a34a;
}
.r2r-step-active {
    border-color: #fb923c;
    background: #fff7ed;
}
.r2r-step-active .r2r-step-dot {
    background: #f97316;
    box-shadow: 0 0 0 6px rgba(249, 115, 22, 0.18);
}
.r2r-step-waiting {
    color: #64748b;
}
.r2r-answer-card {
    min-height: 236px;
    border: 1px solid #d8dee8;
    border-radius: 8px;
    background: #ffffff;
    padding: 1.05rem 1.1rem;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
}
.r2r-answer-card h3 {
    margin: 0 0 0.75rem;
    color: #172033;
    font-size: 1.15rem;
}
.r2r-answer-card p {
    margin: 0 0 0.8rem;
    color: #334155;
    font-size: 0.98rem;
    line-height: 1.55;
}
.r2r-answer-card p:last-child {
    margin-bottom: 0;
}
.r2r-answer-card strong {
    color: #0f172a;
}
.r2r-answer-card code {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    color: #0f172a;
    padding: 0.08rem 0.28rem;
}
.r2r-verdict {
    border-radius: 8px;
    display: grid;
    gap: 0.2rem;
    margin: 0 0 0.9rem;
    padding: 0.75rem 0.85rem;
}
.r2r-verdict span {
    color: #0f172a;
    font-size: 1rem;
    font-weight: 800;
}
.r2r-verdict small {
    color: #475569;
    font-size: 0.86rem;
    line-height: 1.35;
}
.r2r-verdict-caution {
    background: #fff7ed;
    border: 1px solid #fed7aa;
}
.r2r-confidence {
    align-items: start;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    display: grid;
    gap: 0.2rem;
    margin: 0 0 0.85rem;
    padding: 0.65rem 0.8rem;
}
.r2r-confidence strong {
    color: #172033;
}
.r2r-confidence span {
    color: #475569;
    font-size: 0.88rem;
}
.r2r-output-list {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    margin: 0 0 0.85rem;
    padding: 0.75rem 0.85rem;
}
.r2r-output-list strong {
    color: #0f172a !important;
    display: block;
    font-size: 0.98rem;
    margin-bottom: 0.4rem;
    opacity: 1 !important;
}
.r2r-output-list ul {
    color: #172033 !important;
    margin: 0;
    padding-left: 1.1rem;
    opacity: 1 !important;
}
.r2r-output-list li {
    color: #172033 !important;
    font-size: 0.92rem;
    line-height: 1.45;
    margin: 0.25rem 0;
    opacity: 1 !important;
}
.r2r-output-list li::marker {
    color: #ea580c !important;
}
.r2r-limitations {
    background: #fff7ed;
    border-color: #fdba74;
}
.r2r-limitations li,
.r2r-limitations ul {
    color: #3f2a12 !important;
}
.r2r-limitations strong {
    color: #7c2d12 !important;
}
.r2r-verdict-supported {
    background: #ecfdf5;
    border: 1px solid #bbf7d0;
}
.r2r-verdict-refuted {
    background: #fef2f2;
    border: 1px solid #fecaca;
}
.r2r-verdict-mixed {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
}
.r2r-answer-ready {
    background:
        linear-gradient(135deg, rgba(255, 247, 237, 0.82), rgba(255, 255, 255, 0.96)),
        #ffffff;
}
.r2r-answer-working {
    border-color: #fb923c;
    background: #fff7ed;
}
.r2r-answer-warning {
    border-color: #f97316;
    background: #fff7ed;
}
.r2r-section-title {
    color: #334155;
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 0;
    margin: 0 0 0.5rem;
}
.r2r-note {
    border-left: 4px solid #f97316;
    color: #475569;
    background: #fff7ed;
    padding: 0.75rem 0.9rem;
    border-radius: 6px;
    margin-bottom: 1rem;
}
.r2r-note strong {
    color: #172033;
}
.r2r-empty-evidence {
    align-items: center;
    background: rgba(255, 255, 255, 0.78);
    border: 1px dashed #cbd5e1;
    border-radius: 8px;
    color: #64748b;
    display: flex;
    gap: 0.55rem;
    min-height: 64px;
    padding: 1rem;
}
.r2r-empty-evidence strong {
    color: #334155;
}
.r2r-sources-grid {
    display: grid;
    gap: 0.75rem;
    grid-template-columns: repeat(2, minmax(0, 1fr));
}
.r2r-source-card {
    background: #ffffff;
    border: 1px solid #d8dee8;
    border-radius: 8px;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    min-height: 150px;
    padding: 0.95rem;
}
.r2r-source-topline {
    align-items: center;
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 0.7rem;
}
.r2r-source-rank,
.r2r-source-score {
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 0.22rem 0.5rem;
}
.r2r-source-rank {
    background: #172033;
    color: #ffffff;
}
.r2r-source-score {
    background: #ecfdf5;
    color: #047857;
}
.r2r-source-card h3 {
    color: #172033;
    font-size: 0.98rem;
    line-height: 1.3;
    margin: 0 0 0.55rem;
}
.r2r-source-card h3 span {
    color: #64748b;
    font-weight: 600;
}
.r2r-source-card p {
    color: #475569;
    font-size: 0.9rem;
    line-height: 1.45;
    margin: 0 0 0.75rem;
}
.r2r-source-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin: 0 0 0.65rem;
}
.r2r-source-meta span {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 999px;
    color: #334155;
    font-size: 0.76rem;
    font-weight: 700;
    padding: 0.18rem 0.48rem;
}
.r2r-source-card blockquote {
    border-left: 3px solid #fed7aa;
    color: #475569;
    font-size: 0.84rem;
    line-height: 1.4;
    margin: 0 0 0.75rem;
    padding-left: 0.65rem;
}
.r2r-source-card a {
    color: #ea580c;
    font-weight: 700;
    text-decoration: none;
}
@media (max-width: 820px) {
    .r2r-sources-grid {
        grid-template-columns: 1fr;
    }
}
button.primary {
    min-height: 46px !important;
}
footer {
    display: none !important;
}
"""


with gr.Blocks(
    title="Rumor2Reality",
    css=CSS,
    theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
) as demo:
    with gr.Column(elem_classes=["r2r-shell"]):
        gr.Markdown(
            """
            <div class="r2r-title">
              <h1>Rumor2Reality</h1>
              <p>Evidence-grounded rumor checking with visible retrieval, citations, and latency.</p>
              <div class="r2r-badges">
                <span class="r2r-badge">RAG pipeline</span>
                <span class="r2r-badge">Source transparency</span>
                <span class="r2r-badge">Demo-ready deploy</span>
              </div>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                gr.Markdown(
                    """
                    <div class="r2r-note">
                      <strong>Demo flow:</strong> paste a claim, run the check, then explain how the system moved from papers to evidence to final answer.
                    </div>
                    """
                )
                claim_input = gr.Textbox(
                    label="Rumor or claim",
                    value=DEFAULT_CLAIM,
                    lines=6,
                    max_lines=8,
                    placeholder="Paste a rumor, viral claim, or challenge prompt...",
                )
            with gr.Column(scale=2):
                model_name = gr.Dropdown(
                    label="Answer mode",
                    choices=["starter-rag-baseline", "local-llm", "api-llm"],
                    value="starter-rag-baseline",
                )
                top_k = gr.Slider(
                    label="Number of papers",
                    minimum=1,
                    maximum=5,
                    value=4,
                    step=1,
                )
                run_button = gr.Button("Run check", variant="primary")
                clear_button = gr.Button("Clear")

        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                gr.Markdown('<p class="r2r-section-title">Pipeline status</p>')
                progress = gr.HTML(value=_progress_html("ready"))
            with gr.Column(scale=3):
                gr.Markdown('<p class="r2r-section-title">Assessment</p>')
                answer = gr.HTML(value=_answer_html(READY_ANSWER))

        gr.Markdown('<p class="r2r-section-title">Retrieved evidence</p>')
        sources = gr.HTML(value=_empty_sources_html())

        with gr.Accordion("Debug contract for backend integration", open=False):
            debug = gr.Code(label="Pipeline output", language="json")

        with gr.Accordion("Timing and metadata", open=False):
            timing = gr.JSON(label="Run details", value=READY_TIMING)

        with gr.Accordion("Example challenge prompts", open=False):
            gr.Examples(
                examples=[
                    [DEFAULT_CLAIM, 4, "starter-rag-baseline"],
                    ["A thread claims that AI-generated misinformation spreads faster than human-written rumors.", 5, "starter-rag-baseline"],
                    ["People say retrieval augmented generation always prevents hallucinations.", 3, "starter-rag-baseline"],
                ],
                inputs=[claim_input, top_k, model_name],
            )

    run_event = run_button.click(
        analyse_claim,
        inputs=[claim_input, top_k, model_name],
        outputs=[progress, answer, sources, timing, debug],
    )
    claim_input.submit(
        analyse_claim,
        inputs=[claim_input, top_k, model_name],
        outputs=[progress, answer, sources, timing, debug],
    )
    clear_button.click(
        clear_outputs,
        outputs=[progress, answer, sources, timing, debug],
        cancels=[run_event],
    )


if __name__ == "__main__":
    demo.queue().launch()
