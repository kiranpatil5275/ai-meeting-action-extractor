"""
Streamlit UI for AI Meeting Action-Item Extractor
Improved design: modern layout, sidebar controls, color-coded table, metrics, exports.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.extractor import ActionItemExtractor
from src.validator import ActionItemValidator

# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Meeting Action-Item Extractor",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
    }
    .stButton > button[kind="primary"] {
        border-radius: 8px;
        font-weight: 600;
    }
    h2, h3 { color: #0f172a; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stCheckbox label {
        color: #94a3b8 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_extractor(
    confidence_threshold: float,
    owner_boost: float,
    deadline_boost: float,
    rule_boost: float,
    use_zero_shot: bool,
):
    return ActionItemExtractor(
        confidence_threshold=confidence_threshold,
        owner_boost=owner_boost,
        deadline_boost=deadline_boost,
        rule_boost=rule_boost,
        use_zero_shot=use_zero_shot,
    )


@st.cache_resource
def get_validator(min_confidence: float):
    return ActionItemValidator(min_confidence=min_confidence)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📋 Action-Item Extractor")
    st.caption("Convert meeting transcripts → structured tasks")
    st.markdown("---")

    st.subheader("⚙️ Extraction Settings")
    use_zero_shot = st.checkbox(
        "Use zero-shot model (BART-MNLI)",
        value=True,
        help="Uncheck for pure rule-based mode (faster, no model download)",
    )
    conf_threshold = st.slider("Confidence threshold", 0.20, 0.80, 0.40, 0.05)
    owner_boost = st.slider("Owner boost", 0.0, 0.25, 0.12, 0.01)
    deadline_boost = st.slider("Deadline boost", 0.0, 0.25, 0.12, 0.01)
    rule_boost = st.slider("Rule-pattern boost", 0.0, 0.20, 0.08, 0.01)
    min_conf_display = st.slider("Highlight below confidence", 0.20, 0.70, 0.45, 0.05)

    st.markdown("---")
    st.subheader("📂 Sample Transcripts")
    sample_dir = Path(__file__).parent / "data" / "sample_transcripts"
    sample_files = sorted(sample_dir.glob("transcript_*.txt")) if sample_dir.exists() else []
    selected_sample = st.selectbox(
        "Load a sample",
        options=["(none)"] + [f.name for f in sample_files],
        index=0,
    )

    if selected_sample != "(none)":
        sample_text = (sample_dir / selected_sample).read_text(encoding="utf-8")
    else:
        sample_text = ""

    st.markdown("---")
    st.caption(
        "Models: spaCy + facebook/bart-large-mnli  \n"
        "Fully offline after first download  \n"
        "20 sample transcripts available"
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("AI Meeting Action-Item Extractor")
st.markdown(
    "Upload a transcript or paste text. The system extracts **task · owner · deadline · confidence** "
    "and flags incomplete or low-confidence items."
)

tab_upload, tab_paste, tab_sample = st.tabs(["📤 Upload", "📝 Paste", "📚 Sample"])

transcript = ""
with tab_upload:
    uploaded = st.file_uploader("Drop a .txt transcript", type=["txt"], label_visibility="collapsed")
    if uploaded is not None:
        transcript = uploaded.read().decode("utf-8")
        st.success(f"Loaded **{uploaded.name}** ({len(transcript)} chars)")

with tab_paste:
    pasted = st.text_area(
        "Paste transcript",
        height=260,
        placeholder="Speaker Name: Let's discuss the action items...\n...",
        label_visibility="collapsed",
    )
    if pasted.strip():
        transcript = pasted

with tab_sample:
    if sample_text:
        st.code(sample_text[:900] + ("…" if len(sample_text) > 900 else ""), language=None)
        if st.button("Use this sample", key="use_sample_btn"):
            st.session_state["force_sample"] = sample_text
            transcript = sample_text
    else:
        st.info("Select a sample from the sidebar.")

if st.session_state.get("force_sample"):
    transcript = st.session_state["force_sample"]

col_btn, col_info = st.columns([1, 3])
with col_btn:
    extract_btn = st.button("🚀 Extract Action Items", type="primary", use_container_width=True)
with col_info:
    if not transcript.strip():
        st.caption("Provide a transcript via Upload, Paste, or Sample first.")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if extract_btn and transcript.strip():
    with st.spinner("Extracting action items… (first run may download the zero-shot model ~1.6 GB)"):
        extractor = get_extractor(
            confidence_threshold=conf_threshold,
            owner_boost=owner_boost,
            deadline_boost=deadline_boost,
            rule_boost=rule_boost,
            use_zero_shot=use_zero_shot,
        )
        validator = get_validator(min_confidence=min_conf_display)

        items = extractor.extract(transcript)
        items = validator.validate(items)
        summary = validator.summary(items)

    st.markdown("### Results Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Items", summary["total"])
    pct = f"{100 * summary['complete'] / summary['total']:.0f}%" if summary["total"] else None
    m2.metric("Complete", summary["complete"], delta=pct)
    m3.metric("Incomplete", summary["incomplete"])
    issue_count = sum(summary["issues"].values()) if summary["issues"] else 0
    m4.metric("Issues Flagged", issue_count)

    if summary["issues"]:
        chips = " · ".join(f"**{k}**: {v}" for k, v in summary["issues"].items())
        st.info(f"Issue breakdown → {chips}")

    rows = []
    for i, item in enumerate(items, 1):
        rows.append({
            "#": i,
            "Task": item.task,
            "Owner": item.owner or "—",
            "Deadline": item.deadline or "—",
            "Status": item.status.capitalize(),
            "Confidence": item.confidence,
            "Issues": ", ".join(item.issues) if item.issues else "—",
            "Source": (item.source_sentence[:100] + "…") if len(item.source_sentence) > 100 else item.source_sentence,
        })
    df = pd.DataFrame(rows)

    st.markdown("### Extracted Action Items")
    if df.empty:
        st.warning("No action items detected. Try lowering the confidence threshold or check the transcript format.")
    else:
        def highlight_row(row):
            styles = [""] * len(row)
            issues = str(row.get("Issues", ""))
            conf = float(row.get("Confidence", 1.0))
            if any(x in issues for x in ("missing_owner", "missing_deadline", "empty_or_short_task")):
                styles = ["background-color: #fef3c7"] * len(row)
            elif "low_confidence" in issues or conf < min_conf_display:
                styles = ["background-color: #fee2e2"] * len(row)
            elif "duplicate" in issues:
                styles = ["background-color: #e2e8f0"] * len(row)
            elif conf >= 0.75 and issues in ("—", ""):
                styles = ["background-color: #d1fae5"] * len(row)
            return styles

        styled = (
            df.style
            .apply(highlight_row, axis=1)
            .format({"Confidence": "{:.2f}"})
        )
        st.dataframe(styled, use_container_width=True, height=min(480, 60 + 35 * len(df)))

        st.caption(
            "🟢 High confidence & complete  ·  "
            "🟡 Missing owner / deadline  ·  "
            "🔴 Low confidence  ·  "
            "⬜ Duplicate"
        )

        st.markdown("### Export")
        c1, c2, c3 = st.columns(3)
        with c1:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ CSV", data=csv, file_name="action_items.csv", mime="text/csv", use_container_width=True)
        with c2:
            json_bytes = json.dumps([item.to_dict() for item in items], indent=2).encode("utf-8")
            st.download_button("⬇️ JSON", data=json_bytes, file_name="action_items.json", mime="application/json", use_container_width=True)
        with c3:
            md_lines = ["# Action Items\n"]
            for item in items:
                md_lines.append(
                    f"- **{item.task}**  \n  Owner: {item.owner or '—'} · "
                    f"Deadline: {item.deadline or '—'} · Conf: {item.confidence:.2f}"
                )
            md = "\n".join(md_lines).encode("utf-8")
            st.download_button("⬇️ Markdown", data=md, file_name="action_items.md", mime="text/markdown", use_container_width=True)

elif extract_btn:
    st.warning("Please provide a transcript first.")

st.markdown("---")
st.caption(
    "Built with spaCy + Hugging Face Transformers (zero-shot) + Streamlit · "
    "Fully open-source & offline-capable after initial model download · "
    "20 annotated sample transcripts included"
)
