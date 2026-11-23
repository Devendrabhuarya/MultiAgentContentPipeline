"""
Streamlit dashboard for the Multi-Agent Content Pipeline.

Run with:
    streamlit run ui/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running via `streamlit run ui/streamlit_app.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from app.config import settings
from app.graph.workflow import run_pipeline
from app.llm.client import LLMError, MockLLMClient, OpenAICompatibleLLMClient
from app.models.schemas import ContentType, UserRequest
from app.research import get_research_provider
from app.research.mock import MockResearchProvider

st.set_page_config(page_title="Multi-Agent Content Pipeline", layout="wide")

STEP_LABELS = [
    "Research Agent", "Planning Agent", "Writing Agent", "Fact Checker",
    "SEO Agent", "Critic Agent", "Feedback Agent",
]

st.title("🧠 Multi-Agent AI Content Pipeline")
st.caption("Research → Planning → Writing → Fact Check → SEO → Critic → Feedback → Revision → Final")

with st.sidebar:
    st.header("Content Requirements")
    topic = st.text_input("Topic", value="Federated Learning in Healthcare")
    content_type = st.selectbox("Content type", [c.value for c in ContentType])
    audience = st.text_input("Target audience", value="beginner-friendly general audience")
    tone = st.text_input("Tone", value="professional")
    word_count = st.number_input("Desired word count", min_value=100, max_value=5000, value=1200, step=100)
    keywords_raw = st.text_input("Keywords (comma-separated)", value="federated learning, healthcare AI")
    include_refs = st.checkbox("Include references", value=True)
    max_iterations = st.slider("Maximum iterations", 0, 10, settings.max_iterations)
    quality_threshold = st.slider("Quality threshold", 0, 100, int(settings.quality_threshold))

    st.divider()
    use_mock = st.checkbox(
        "Use offline mock mode (no API key needed)",
        value=not bool(settings.openai_api_key),
        help="Runs the full graph with a deterministic mock LLM + mock research "
        "provider. Useful for demoing the architecture without API costs.",
    )
    run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)


def _mock_client() -> MockLLMClient:
    """A MockLLMClient pre-loaded with a plausible response sequence so the
    UI demo shows the score improving across iterations, exactly like a
    real run would."""
    import json

    responses = [
        json.dumps({
            "topic": topic, "key_points": ["Point A", "Point B"], "facts": ["Fact 1", "Fact 2"],
            "sources": [], "important_claims": ["Claim 1"], "uncertainties": [],
            "subtopics": ["Sub A", "Sub B"],
        }),
        json.dumps({
            "title": f"{topic}: A Comprehensive Overview", "introduction_plan": "Intro",
            "sections": [{"heading": "Background", "subsections": [], "key_points_to_cover": ["Point A"],
                          "evidence_refs": [], "estimated_word_count": 400, "requires_citation": True}],
            "conclusion_plan": "Wrap up", "evidence_mapping": {}, "estimated_total_word_count": 400,
        }),
    ]
    # Draft v1 (lower quality) -> feedback -> draft v2 (higher quality) -> final
    for score, wc in [(62, 400), (86, 420)]:
        responses.append(json.dumps({
            "title": topic, "body_markdown": f"# {topic}\n\nDraft content ({score=}).",
            "meta_description": "desc", "word_count": wc, "references": [], "version": 1,
        }))
        responses.append(json.dumps({
            "claims_checked": 10, "supported": 8 if score < 80 else 10, "unsupported": 2 if score < 80 else 0,
            "uncertain": 0, "contradicted": 0, "issues": [], "confidence_score": score,
        }))
        responses.append(json.dumps({
            "seo_score": score, "detected_keywords": ["federated learning"], "missing_keywords": [],
            "title_assessment": "ok", "heading_assessment": "ok", "readability_assessment": "ok",
            "meta_description_assessment": "ok", "recommendations": ["Improve keyword density"],
        }))
        responses.append(json.dumps({
            "factuality": score, "relevance": score, "quality": score, "structure": score,
            "readability": score, "seo": score, "evidence": score, "requirement_compliance": score,
            "overall_score": score, "strengths": ["Clear intro"], "weaknesses": ["Needs more detail"],
            "critical_issues": [] if score >= 80 else ["Add supporting evidence"],
            "minor_issues": ["Tighten conclusion"], "revision_recommendations": ["Add a statistic"],
        }))
        if score < 80:
            responses.append(json.dumps({
                "priority": ["Add supporting evidence", "Tighten conclusion"],
                "instructions": ["Add a cited statistic to the background section."],
                "critical_fixes": ["Add supporting evidence"], "cosmetic_fixes": ["Tighten conclusion"],
            }))
    return MockLLMClient(responses=responses)


if run_btn:
    request = UserRequest(
        topic=topic,
        content_type=ContentType(content_type),
        target_audience=audience,
        tone=tone,
        word_count=word_count,
        keywords=[k.strip() for k in keywords_raw.split(",") if k.strip()],
        include_references=include_refs,
        max_iterations=max_iterations,
        quality_threshold=float(quality_threshold),
    )

    progress_box = st.empty()
    with st.spinner("Running multi-agent pipeline..."):
        try:
            if use_mock:
                llm_client = _mock_client()
                research_provider = MockResearchProvider()
            else:
                llm_client = OpenAICompatibleLLMClient()
                research_provider = get_research_provider()

            state = run_pipeline(request, llm_client, research_provider)
        except LLMError as exc:
            st.error(f"LLM error: {exc}")
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Pipeline failed: {exc}")
            st.stop()

    final = state["final_content"]
    history = state.get("revision_history", [])

    st.success(
        f"Pipeline completed in {final.total_execution_time_seconds}s "
        f"across {final.iterations_used} revision(s). Status: {final.status.value}"
    )

    # --- Workflow progress ---
    st.subheader("Workflow Progress")
    cols = st.columns(len(STEP_LABELS))
    for col, label in zip(cols, STEP_LABELS):
        col.markdown(f"**{label}**  \n✅")
    for i in range(1, len(history)):
        st.markdown(f"✅ Revision {i} — Critic re-evaluation ✅")

    # --- Final content ---
    st.subheader("📄 Final Content")
    st.markdown(f"### {final.final_content.title}")
    st.markdown(final.final_content.body_markdown)
    if final.final_content.references:
        st.markdown("**References**")
        for ref in final.final_content.references:
            st.markdown(f"- {ref}")

    # --- Quality score ---
    st.subheader("📊 Quality Score")
    qs = final.quality_score
    score_cols = st.columns(4)
    score_cols[0].metric("Overall", f"{qs.overall_score:.0f}")
    score_cols[1].metric("Factuality", f"{qs.factuality:.0f}")
    score_cols[2].metric("Relevance", f"{qs.relevance:.0f}")
    score_cols[3].metric("Structure", f"{qs.structure:.0f}")
    score_cols2 = st.columns(4)
    score_cols2[0].metric("Readability", f"{qs.readability:.0f}")
    score_cols2[1].metric("SEO", f"{qs.seo:.0f}")
    score_cols2[2].metric("Evidence", f"{qs.evidence:.0f}")
    score_cols2[3].metric("Requirement Compliance", f"{qs.requirement_compliance:.0f}")

    if final.warnings:
        for w in final.warnings:
            st.warning(w)

    # --- Revision history ---
    st.subheader("🔁 Revision History")
    if history:
        df = pd.DataFrame(
            {
                "Version": [r.version for r in history],
                "Overall Score": [r.critic_evaluation.overall_score for r in history],
                "Factuality": [r.critic_evaluation.factuality for r in history],
                "SEO": [r.critic_evaluation.seo for r in history],
                "Word Count": [r.draft.word_count for r in history],
            }
        )
        st.line_chart(df.set_index("Version")["Overall Score"])
        st.dataframe(df, use_container_width=True, hide_index=True)

    # --- Feedback history ---
    st.subheader("💬 Feedback History")
    for r in history:
        if r.feedback:
            with st.expander(f"Feedback after version {r.version}"):
                st.write("**Priority fixes:**")
                for p in r.feedback.priority:
                    st.write(f"- {p}")
                st.write("**Instructions given to writer:**")
                for ins in r.feedback.instructions:
                    st.write(f"- {ins}")
    if not any(r.feedback for r in history):
        st.caption("No revisions were needed — first draft met the quality threshold.")

else:
    st.info("Configure your content requirements in the sidebar and click **Run Pipeline**.")
