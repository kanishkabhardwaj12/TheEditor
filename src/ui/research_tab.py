"""Research assistant tab UI."""

from html import escape
from urllib.parse import urlparse

import streamlit as st

from src.research.brief_generator import BriefGenerationError, generate_research_brief
from src.research.extractor import extract_sources
from src.research.ingest import parse_sources
from src.research.source_summarizer import SourceSummarizationError, summarize_sources


def _render_research_styles() -> None:
    st.markdown(
        """
        <style>
        .research-tab-section { margin: 1.15rem 0 1.4rem; }
        .research-tab-section h3 { margin-bottom: .35rem; }
        .research-tab-kicker {
            color: #8ea8ff; font-size: .73rem; font-weight: 700;
            letter-spacing: .08em; text-transform: uppercase;
        }
        .research-tab-lead {
            border-left: 3px solid #7c9cff; padding: .3rem 0 .3rem 1rem;
            color: #d7deea; line-height: 1.7; font-size: 1.04rem;
        }
        .research-tab-finding {
            border-left: 3px solid #f0c674; padding: .55rem 0 .55rem 1rem;
            color: #f3f6fb; line-height: 1.65; font-size: 1.02rem;
        }
        .research-tab-source, .research-tab-evidence {
            border-top: 1px solid rgba(145, 167, 216, .25);
            padding: .85rem 0; margin-top: .45rem;
        }
        .research-tab-source-title, .research-tab-claim {
            color: #f3f6fb; font-weight: 650; line-height: 1.45;
        }
        .research-tab-source-meta, .research-tab-attribution {
            color: #9da9bc; font-size: .82rem; margin-top: .3rem; line-height: 1.5;
        }
        .research-tab-excerpt {
            color: #c1cad9; font-style: italic; line-height: 1.55; margin-top: .45rem;
        }
        .research-tab-status {
            border-bottom: 1px solid rgba(145, 167, 216, .2);
            padding: .65rem 0;
        }
        .research-tab-status strong { color: #f3f6fb; }
        .research-tab-angle {
            border-left: 2px solid #8ea8ff; padding: .15rem 0 .15rem 1rem;
            margin: .5rem 0 1.3rem; height: 100%;
        }
        .research-tab-angle h4 { color: #f6d88b; margin: 0 0 .55rem; }
        .research-tab-label { color: #8ea8ff; font-weight: 650; }
        .research-tab-empty {
            color: #9da9bc; font-style: italic; padding: .35rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _safe_link(url: object, label: object) -> str:
    """Return a link only for an absolute HTTP(S) URL."""
    candidate = _text(url)
    title = escape(_text(label) or candidate)
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return title
    return (
        f'<a href="{escape(candidate, quote=True)}" target="_blank" '
        f'rel="noopener noreferrer">{title}</a>'
    )


def _items(items: object) -> str:
    values = items if isinstance(items, (list, tuple)) else []
    rendered = "".join(
        f"<li>{escape(_text(item))}</li>" for item in values if _text(item)
    )
    return rendered or '<li class="research-tab-empty">No items available.</li>'


def _section_heading(title: str, kicker: str = "") -> None:
    if kicker:
        st.markdown(
            f'<div class="research-tab-kicker">{escape(kicker)}</div>',
            unsafe_allow_html=True,
        )
    st.subheader(title)


def _paragraph_section(title: str, value: object, kicker: str = "") -> None:
    _section_heading(title, kicker)
    content = _text(value) or "No synthesis is available for this section."
    st.markdown(
        f'<div class="research-tab-lead">{escape(content)}</div>',
        unsafe_allow_html=True,
    )


def _list_section(title: str, values: object, kicker: str = "") -> None:
    _section_heading(title, kicker)
    st.markdown(f"<ul>{_items(values)}</ul>", unsafe_allow_html=True)


def _render_source_status(source: object) -> None:
    status_value = _text(getattr(source, "status", "failed"))
    labels = {
        "extracted": ("Extracted", "#74d99a"),
        "partially_extracted": ("Partially extracted", "#f0c674"),
        "blocked": ("Blocked", "#f0c674"),
        "failed": ("Failed", "#f08080"),
    }
    status, color = labels.get(status_value, ("Unavailable", "#f08080"))
    label = (
        _text(getattr(source, "title", ""))
        or _text(getattr(source, "url", ""))
        or "Pasted source"
    )
    method = _text(getattr(source, "extraction_method", ""))
    words = getattr(source, "word_count", 0) or 0
    error = _text(getattr(source, "error", ""))
    detail = f"via {method} · {words:,} words" if method else f"{words:,} words"
    if error:
        detail = f"{detail} · {error}"
    st.markdown(
        f'<div class="research-tab-status"><strong>{escape(label)}</strong> '
        f'<span style="color:{color};float:right;font-weight:700">{status}</span>'
        f'<div class="research-tab-source-meta">{escape(detail)}</div></div>',
        unsafe_allow_html=True,
    )


def _render_source_finding(finding: object) -> None:
    title = _text(getattr(finding, "title", "")) or "Untitled source"
    source_id = _text(getattr(finding, "source_id", ""))
    source_label = _safe_link(getattr(finding, "url", None), title)
    claim = _text(getattr(finding, "claim", "")) or "No main finding available."
    metadata = []
    for label, field in (
        ("Method", "methodology"),
        ("Sample", "sample_size"),
        ("Population", "population"),
        ("Date", "date"),
    ):
        value = _text(getattr(finding, field, ""))
        if value:
            metadata.append(f"{label}: {escape(value)}")
    meta = " · ".join(metadata)
    if source_id:
        meta = f"{meta}{' · ' if meta else ''}Source ID: {escape(source_id)}"
    st.markdown(
        f'<div class="research-tab-source">'
        f'<div class="research-tab-source-title">{source_label}</div>'
        f'<div class="research-tab-claim">{escape(claim)}</div>'
        f'{f"<div class=research-tab-source-meta>{meta}</div>" if meta else ""}'
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_evidence(item: object) -> None:
    claim = _text(getattr(item, "claim", "")) or "Unlabeled evidence"
    excerpt = _text(getattr(item, "excerpt", ""))
    title = _text(getattr(item, "title", "")) or "Untitled source"
    source_id = _text(getattr(item, "source_id", ""))
    attribution = _safe_link(getattr(item, "url", None), title)
    if source_id:
        attribution += f" · {escape(source_id)}"
    excerpt_html = (
        f'<div class="research-tab-excerpt">“{escape(excerpt)}”</div>' if excerpt else ""
    )
    st.markdown(
        f'<div class="research-tab-evidence">'
        f'<div class="research-tab-claim">{escape(claim)}</div>'
        f"{excerpt_html}"
        f'<div class="research-tab-attribution">{attribution}</div></div>',
        unsafe_allow_html=True,
    )


def _render_angle(angle: object, recommended: bool = False) -> None:
    title = _text(getattr(angle, "title", "")) or "Untitled angle"
    marker = "Recommended · " if recommended else ""
    fields = (
        ("Audience", "audience"),
        ("Central argument", "central_argument"),
        ("Why it works", "why_interesting"),
        ("Rationale", "rationale"),
        ("Further reporting", "reporting_needed"),
    )
    body = []
    for label, field in fields:
        value = _text(getattr(angle, field, ""))
        if value:
            body.append(
                f'<p><span class="research-tab-label">{label}</span><br>'
                f"{escape(value)}</p>"
            )
    body_html = "".join(body) or (
        '<p class="research-tab-empty">No detail available.</p>'
    )
    st.markdown(
        f'<div class="research-tab-angle"><h4>{escape(marker + title)}</h4>'
        f"{body_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_research_tab() -> None:
    st.subheader("Research Assistant")
    st.write("Collect sources and generate an evidence-backed editorial brief.")
    _render_research_styles()

    urls_text = st.text_area(
        "Source URLs (one per line)",
        placeholder="https://...\nhttps://...\nhttps://...",
        help="Provide at least 3 readable sources. Only HTTP and HTTPS URLs are accepted.",
        height=140,
    )
    pasted_text = st.text_area(
        "Optional pasted source text",
        help=(
            "Blank lines are normal paragraph breaks. Separate multiple pasted sources with "
            "a line containing exactly ---SOURCE---."
        ),
        placeholder="Paste one complete article or source here...\n\n---SOURCE---",
        height=160,
    )

    if not st.button("Generate research brief", type="primary"):
        return

    sources = parse_sources(urls_text=urls_text, pasted_text=pasted_text)
    if len(sources) < 3:
        st.error("Please provide at least 3 sources (URLs or pasted entries).")
        return

    st.info("Extracting and validating each source...")
    extracted_sources = extract_sources(sources)
    _section_heading("Source status")
    for source in extracted_sources:
        _render_source_status(source)

    usable_sources = [
        source
        for source in extracted_sources
        if source.status in {"extracted", "partially_extracted"}
        and source.content.strip()
        and source.word_count >= 25
    ]
    if len(usable_sources) < 3:
        st.error(
            "At least 3 sources must be readable. Fix blocked or failed URLs, or paste their "
            "text manually, then try again."
        )
        return

    progress = st.empty()
    try:
        progress.info("Analyzing readable sources...")

        def update_progress(completed: int, total: int, message: str) -> None:
            progress.info(f"{message} ({completed}/{total})")

        summaries = summarize_sources(usable_sources, update_progress)
        progress.info("Source analysis complete. Synthesizing the editorial brief...")
        brief = generate_research_brief(summaries)
        progress.success("Research brief generated.")
    except (SourceSummarizationError, BriefGenerationError) as exc:
        progress.empty()
        st.error(str(exc))
        return

    missing_sections = getattr(brief, "missing_sections", None) or []
    if missing_sections:
        st.warning(
            "Synthesis incomplete. Missing sections: "
            + ", ".join(missing_sections)
            + ". Source evidence and verified facts are still available below."
        )

    _paragraph_section("What Happened", brief.what_happened, "What the sources report")
    _paragraph_section(
        "Most Significant Finding",
        brief.most_significant_finding,
        "Cross-source signal",
    )
    _paragraph_section("Why It Matters", brief.why_it_matters, "Practical relevance")
    _list_section("Who Should Care", brief.who_should_care, "Audience")

    _section_heading("What the Sources Say")
    findings = getattr(brief, "source_findings", None) or []
    if findings:
        for finding in findings:
            _render_source_finding(finding)
    else:
        st.markdown(
            '<div class="research-tab-empty">No source-level findings are available.</div>',
            unsafe_allow_html=True,
        )

    _list_section("Source Comparison", brief.source_comparison, "Agreements and gaps")
    _list_section("Source-attributed Claims", brief.verified_facts, "Traceable claims")

    _section_heading("Analysis and Implications")
    analysis = brief.analysis or []
    implications = getattr(brief, "implications", None) or []
    left, right = st.columns(2)
    with left:
        st.markdown("**Analysis**")
        st.markdown(f"<ul>{_items(analysis)}</ul>", unsafe_allow_html=True)
    with right:
        st.markdown("**Practical implications**")
        st.markdown(f"<ul>{_items(implications)}</ul>", unsafe_allow_html=True)

    _section_heading("Speculation and Reporting Gaps")
    speculation, questions = brief.speculation or [], brief.open_questions or []
    left, right = st.columns(2)
    with left:
        st.markdown("**Speculation**")
        st.markdown(f"<ul>{_items(speculation)}</ul>", unsafe_allow_html=True)
    with right:
        st.markdown("**Open questions**")
        st.markdown(f"<ul>{_items(questions)}</ul>", unsafe_allow_html=True)

    _section_heading("Evidence Trail", "Exact supporting excerpts")
    records = getattr(brief, "evidence_records", None) or []
    if records:
        for item in records:
            _render_evidence(item)
    else:
        st.markdown(
            '<div class="research-tab-empty">No evidence records are available.</div>',
            unsafe_allow_html=True,
        )
    for item in brief.evidence or []:
        if _text(item):
            st.markdown(f"- {escape(_text(item))}")

    _section_heading("Newsletter Angles", "Editorial directions")
    angles = getattr(brief, "angle_details", None) or []
    if angles:
        columns = st.columns(min(2, len(angles)))
        recommended = _text(getattr(brief, "recommended_angle", ""))
        for index, angle in enumerate(angles):
            with columns[index % len(columns)]:
                _render_angle(angle, _text(getattr(angle, "title", "")) == recommended)
    else:
        st.markdown(
            '<div class="research-tab-empty">No newsletter angles were generated.</div>',
            unsafe_allow_html=True,
        )
    limitation = _text(getattr(brief, "newsletter_angle_limitation", ""))
    if limitation:
        st.info(limitation)
