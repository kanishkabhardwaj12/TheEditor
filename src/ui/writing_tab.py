"""Writing assistant tab UI."""

from html import escape
from collections import Counter

import streamlit as st

from src.writing.analyzer import analyze_draft
from src.writing.editing import (
    AppliedEdit,
    build_current_draft,
    coerce_applied_edit,
    non_overlapping,
    suggestion_key,
    validate_suggestion,
)
from src.writing.metrics import calculate_metrics
from src.writing.ollama_analyzer import OllamaAnalysisError, analyze_with_ollama


def _merge_suggestions(deterministic, llm, text):
    valid = [
        suggestion
        for suggestion in deterministic + llm
        if validate_suggestion(text, suggestion)
    ]
    return non_overlapping(valid)


def _detector_counts(deterministic, llm):
    counts = Counter(item.issue_type for item in deterministic)
    counts.update(item.issue_type for item in llm)
    return dict(counts)


def _highlight_text(text: str, ranges: list[tuple[int, int, str]]) -> str:
    parts = []
    cursor = 0
    for start, end, class_name in sorted(ranges):
        if start < cursor or end > len(text):
            continue
        parts.append(escape(text[cursor:start]))
        parts.append(f'<mark class="{class_name}">{escape(text[start:end])}</mark>')
        cursor = end
    parts.append(escape(text[cursor:]))
    return "".join(parts).replace("\n", "<br>")


def _render_draft_with_highlights(
    label: str, text: str, ranges: list[tuple[int, int, str]]
) -> None:
    st.markdown(f"**{label}**")
    if not text:
        st.caption("Nothing to display.")
        return
    highlighted = _highlight_text(text, ranges)
    st.markdown(
        f'<div class="draft-highlight">{highlighted}</div>',
        unsafe_allow_html=True,
    )


def render_writing_tab() -> None:
    st.subheader("Writing Assistant")
    st.write("Analyze draft clarity, review suggestions, and keep control of edits.")
    st.markdown(
        """
        <style>
        .draft-highlight {
            background: #11141d;
            border: 1px solid #2b3447;
            border-radius: 14px;
            line-height: 1.7;
            margin-bottom: 1rem;
            min-height: 12rem;
            padding: 1rem;
            white-space: normal;
            color: #d7deea;
        }
        .suggestion-highlight {
            background-color: #fff3bf;
            padding: 0.1rem 0.2rem;
        }
        .accepted-highlight {
            background-color: #c6f6d5;
            padding: 0.1rem 0.2rem;
        }
        .writing-panel {
            background: linear-gradient(145deg, #171b26, #11141d);
            border: 1px solid #2b3447;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin: .5rem 0 1rem;
        }
        .writing-panel h4 { margin: 0 0 .45rem; color: #f3f6fb; }
        .issue-chip {
            display: inline-block;
            background: #242b3b;
            border: 1px solid #3a4762;
            border-radius: 999px;
            color: #cbd6ff;
            font-size: .78rem;
            margin: .2rem .25rem .2rem 0;
            padding: .25rem .55rem;
        }
        .suggestion-meta { color: #9da9bc; font-size: .82rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "pending_draft" in st.session_state:
        st.session_state.draft_input = st.session_state.pop("pending_draft")

    draft = st.text_area(
        "Paste your draft",
        placeholder="Paste a newsletter draft here...",
        height=260,
        key="draft_input",
    )

    if "suggestions" not in st.session_state:
        st.session_state.suggestions = []
    if "analyzed_draft" not in st.session_state:
        st.session_state.analyzed_draft = ""
    if "original_draft" not in st.session_state:
        st.session_state.original_draft = ""
    if "analysis_has_run" not in st.session_state:
        st.session_state.analysis_has_run = False
    if "accepted_changes" not in st.session_state:
        st.session_state.accepted_changes = []
    else:
        # Streamlit can retain edits created by an older app version after reload.
        st.session_state.accepted_changes = [
            coerce_applied_edit(item)
            for item in st.session_state.accepted_changes
        ]
    if "rejected_suggestions" not in st.session_state:
        st.session_state.rejected_suggestions = set()

    if st.button("Analyze draft", type="primary"):
        if draft.strip():
            deterministic_suggestions = analyze_draft(draft)
            try:
                with st.spinner("Checking jargon and paragraph clarity with local Qwen..."):
                    llm_suggestions = analyze_with_ollama(draft)
            except OllamaAnalysisError as exc:
                llm_suggestions = []
                st.warning(str(exc))
            st.session_state.suggestions = _merge_suggestions(
                deterministic_suggestions,
                llm_suggestions,
                draft,
            )
            st.session_state.detector_counts = _detector_counts(
                deterministic_suggestions,
                llm_suggestions,
            )
        else:
            st.session_state.suggestions = []
            st.session_state.detector_counts = {}
        st.session_state.analyzed_draft = draft
        st.session_state.original_draft = draft
        st.session_state.analysis_has_run = True
        st.session_state.accepted_changes = []
        st.session_state.rejected_suggestions = set()
    if "detector_counts" not in st.session_state:
        st.session_state.detector_counts = {}

    if draft.strip():
        metrics = calculate_metrics(draft)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Word count", metrics["word_count"])
        c2.metric("Sentences", metrics["sentence_count"])
        c3.metric("Avg sentence length", metrics["avg_sentence_length"])
        c4.metric("Reading time (min)", metrics["reading_time_minutes"])
        st.caption(
            f"Readability hint: {metrics['readability_hint']}. "
            "This is directional, not a definitive quality score."
        )
        st.caption(f"Passive sentences: {metrics['passive_sentence_count']}")

    revised, accepted_ranges = build_current_draft(
        st.session_state.original_draft,
        st.session_state.accepted_changes,
    )
    suggestions = [
        item
        for item in st.session_state.suggestions
        if suggestion_key(item) not in st.session_state.rejected_suggestions
        and not any(
            edit.suggestion_id == suggestion_key(item)
            for edit in st.session_state.accepted_changes
        )
    ]
    categories = {}
    for item in st.session_state.suggestions:
        categories.setdefault(item.issue_type, {}).setdefault(item.severity, 0)
        categories[item.issue_type][item.severity] += 1
    st.markdown("### Review summary")
    st.caption(
        f"Pending: {len(suggestions)} · Accepted: {len(st.session_state.accepted_changes)} · "
        f"Rejected: {len(st.session_state.rejected_suggestions)}"
    )
    if st.session_state.detector_counts:
        st.caption(
            "Detector breakdown: "
            + " · ".join(
                f"{issue_type} {count}"
                for issue_type, count in st.session_state.detector_counts.items()
            )
        )
    if categories:
        chips = "".join(
            f'<span class="issue-chip">{escape(issue_type)} · {sum(severities.values())} '
            f'({escape(", ".join(f"{severity} {count}" for severity, count in severities.items()))})</span>'
            for issue_type, severities in categories.items()
        )
        st.markdown(chips, unsafe_allow_html=True)
    issue_types = ["All"] + sorted({item.issue_type for item in suggestions})
    selected_type = st.selectbox("Filter suggestions", issue_types)
    if selected_type != "All":
        suggestions = [item for item in suggestions if item.issue_type == selected_type]
    if not suggestions:
        if not st.session_state.analysis_has_run:
            st.info("Run analysis to view suggestions.")
            return
        if not draft.strip():
            st.warning("Paste a draft before running the analysis.")
            return
        st.success("Analysis complete. No supported issues were detected.")

    if draft != revised:
        st.warning("The draft changed. Run analysis again to refresh suggestions.")
    editing_allowed = draft == revised

    st.markdown("### Suggestions")
    for idx, suggestion in enumerate(suggestions):
        with st.expander(
            f"{idx + 1}. {suggestion.issue_type} · {suggestion.severity.title()}",
            expanded=idx == 0,
        ):
            st.markdown(
                f'<div class="writing-panel"><h4>Suggested change</h4>'
                f'<p><span class="suggestion-meta">Original passage</span><br>'
                f'{escape(suggestion.original)}</p>'
                f'<p><span class="suggestion-meta">Why this matters</span><br>'
                f'{escape(suggestion.explanation)}</p>'
                f'<p><span class="suggestion-meta">Minimal rewrite</span><br>'
                f'<strong>{escape(suggestion.suggestion)}</strong></p>'
                f'<div class="suggestion-meta">Severity: {escape(suggestion.severity)} · '
                f'Confidence: {suggestion.confidence:.0%} · '
                f'Characters {suggestion.start}–{suggestion.end}</div></div>',
                unsafe_allow_html=True,
            )

            accept_col, reject_col = st.columns(2)
            with accept_col:
                accept = st.button(
                    "Accept",
                    key=f"accept_suggestion_{suggestion_key(suggestion)}",
                    use_container_width=True,
                    disabled=not editing_allowed,
                )
            with reject_col:
                reject = st.button(
                    "Reject",
                    key=f"reject_suggestion_{suggestion_key(suggestion)}",
                    use_container_width=True,
                    disabled=not editing_allowed,
                )

            if accept or reject:
                key = suggestion_key(suggestion)
                if accept:
                    if not validate_suggestion(
                        st.session_state.original_draft, suggestion
                    ):
                        st.error("This suggestion is stale and was not applied. Run analysis again.")
                    else:
                        st.session_state.accepted_changes.append(
                            AppliedEdit(
                                key,
                                suggestion.start,
                                suggestion.end,
                                suggestion.original,
                                suggestion.suggestion,
                            )
                        )
                else:
                    st.session_state.rejected_suggestions.add(key)
                current, _ = build_current_draft(
                    st.session_state.original_draft,
                    st.session_state.accepted_changes,
                )
                st.session_state.pending_draft = current
                st.rerun()

    st.markdown("### Review changes")
    st.caption(
        "Yellow highlights mark passages that received suggestions. "
        "Green highlights mark rewrites you accepted."
    )
    left, right = st.columns(2)
    with left:
        _render_draft_with_highlights(
            "Original draft",
            st.session_state.original_draft,
            [
                (item.start, item.end, "accepted-highlight")
                for item in st.session_state.accepted_changes
            ]
            + [
                (item.start, item.end, "suggestion-highlight")
                for item in suggestions
            ],
        )
    with right:
        _render_draft_with_highlights(
            "Current draft",
            revised,
            [
                (start, end, "accepted-highlight")
                for start, end in accepted_ranges.values()
            ],
        )
