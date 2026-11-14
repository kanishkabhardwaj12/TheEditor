"""Source-grounded research brief generation."""

import json
import logging
import os
from dataclasses import replace
from typing import Any

from src.llm.ollama import OllamaError, chat_json
from src.models.schemas import (
    EvidenceExcerpt,
    NewsletterAngle,
    ResearchBrief,
    SourceFinding,
    SourceSummary,
)


class BriefGenerationError(RuntimeError):
    """Raised when an AI brief cannot be generated or parsed."""


LLM_TIMEOUT_SECONDS = max(120, int(os.getenv("RESEARCH_SYNTHESIS_TIMEOUT_SECONDS", "360")))
LOGGER = logging.getLogger(__name__)
ANGLE_ERROR_PREFIX = "Newsletter angle validation failed:"


def _evidence_id(source_id: str, index: int) -> str:
    return f"{source_id}-evidence-{index + 1}"


def _evidence_entries(summary: SourceSummary) -> list[tuple[str, EvidenceExcerpt]]:
    return [
        (
            evidence.evidence_id or _evidence_id(summary.source_id, index),
            evidence,
        )
        for index, evidence in enumerate(summary.evidence or [])
    ]


def _available_evidence(summaries: list[SourceSummary]) -> list[dict[str, str]]:
    return [
        {
            "evidence_id": evidence_id,
            "source_id": evidence.source_id,
            "claim": evidence.claim,
            "source_reference": (
                f"{evidence.title} ({evidence.url})"
                if evidence.url
                else evidence.title
            ),
        }
        for summary in summaries
        for evidence_id, evidence in _evidence_entries(summary)
    ]


def _source_context(summaries: list[SourceSummary]) -> str:
    entries: list[dict[str, Any]] = []
    for summary in summaries:
        findings = [
            finding.claim
            for finding in (summary.findings or [])
            if finding.claim
        ][:3]
        if not findings:
            findings = [
                claim
                for claim in (summary.verified_facts or [])
                if claim
            ][:3]
        if not findings and (summary.central_argument or summary.topic):
            findings = [summary.central_argument or summary.topic]
        evidence = [
            {
                "evidence_id": evidence_id,
                "claim": item.claim,
                "excerpt": item.excerpt,
            }
            for evidence_id, item in _evidence_entries(summary)[:3]
        ]
        entries.append(
            {
                "source_id": summary.source_id,
                "title": summary.title,
                "url": summary.url or "",
                "topic": summary.central_argument or summary.topic,
                "methodology": summary.methodology,
                "sample_size": summary.sample_size,
                "population": summary.population,
                "date": summary.date,
                "metadata_summary": (
                    f"METHODOLOGY: {summary.methodology} | "
                    f"SAMPLE SIZE: {summary.sample_size} | "
                    f"POPULATION: {summary.population} | DATE: {summary.date}"
                ),
                "findings": findings,
                "evidence": evidence,
                "caveats": [
                    item
                    for item in (summary.caveats or [])
                    if item
                ][:3],
            }
        )
    return json.dumps(entries, ensure_ascii=False, separators=(",", ":"))


def _parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object even when Qwen adds fences or a short preamble."""
    candidate = content.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].lstrip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        payload = None
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(candidate[index:])
                break
            except json.JSONDecodeError:
                continue
        if payload is None:
            raise
    if not isinstance(payload, dict):
        raise json.JSONDecodeError("Expected a JSON object", candidate, 0)
    return payload


def _replace_source_ids(value: str, summaries: list[SourceSummary]) -> str:
    result = value
    for summary in summaries:
        reference = f"{summary.title} ({summary.url})" if summary.url else summary.title
        result = result.replace(summary.source_id, reference)
    return result


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _unique(items: list[str]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        key = " ".join(item.lower().split())
        if key and key not in seen:
            result.append(item)
            seen.add(key)
    return result


def _normalized(value: str) -> str:
    return " ".join(value.lower().split())


def _summary_reference(summary: SourceSummary) -> str:
    return f"{summary.title} ({summary.url})" if summary.url else summary.title


def _source_findings(summaries: list[SourceSummary]) -> list[SourceFinding]:
    findings = []
    for summary in summaries:
        if summary.findings:
            findings.extend(summary.findings)
            continue
        findings.extend(
            SourceFinding(
                source_id=summary.source_id,
                title=summary.title,
                url=summary.url,
                claim=evidence.claim,
                evidence=[evidence],
                methodology=summary.methodology,
                sample_size=summary.sample_size,
                population=summary.population,
                date=summary.date,
                limitation=summary.caveats[0] if summary.caveats else "",
                implication=summary.implications[0] if summary.implications else "",
            )
            for evidence in summary.evidence
        )
    return findings


def _angle_evidence(
    item: dict[str, Any], summaries: list[SourceSummary]
) -> tuple[list[EvidenceExcerpt], int, int]:
    references = item.get("supporting_evidence_ids")
    if not isinstance(references, list):
        legacy_references = item.get("evidence")
        references = (
            [
                reference.get("evidence_id")
                or next(
                    (
                        evidence_id
                        for summary in summaries
                        for evidence_id, evidence in _evidence_entries(summary)
                        if reference.get("source_id") == evidence.source_id
                        and reference.get("excerpt") == evidence.excerpt
                    ),
                    "",
                )
                for reference in legacy_references
                if isinstance(reference, dict)
            ]
            if isinstance(legacy_references, list)
            else []
        )
    matched: list[EvidenceExcerpt] = []
    matched_keys = set()
    valid_ids = {
        evidence_id
        for summary in summaries
        for evidence_id, _ in _evidence_entries(summary)
    }
    invalid_ids = 0
    for summary in summaries:
        for evidence_id, evidence in _evidence_entries(summary):
            if evidence_id in {
                str(reference).strip() for reference in references if reference
            } and evidence_id not in matched_keys:
                matched.append(evidence)
                matched_keys.add(evidence_id)
    invalid_ids = sum(
        1
        for reference in references
        if str(reference).strip() and str(reference).strip() not in valid_ids
    )
    return matched, len(references), invalid_ids


def _parse_angles(
    payload: dict[str, Any],
    summaries: list[SourceSummary],
    strict: bool = True,
    raise_on_failure: bool = True,
) -> tuple[list[NewsletterAngle], dict[str, int]]:
    raw_angles = payload.get("angle_details")
    parsed_count = len(raw_angles) if isinstance(raw_angles, list) else 0
    missing_fields = 0
    duplicate_titles = 0
    evidence_references = 0
    matched_evidence = 0
    invalid_evidence_ids = 0
    angles: list[NewsletterAngle] = []
    seen_titles = set()
    required = (
        "title",
        "audience",
        "central_argument",
        "why_interesting",
        "rationale",
        "reporting_needed",
    )
    for item in raw_angles if isinstance(raw_angles, list) else []:
        if not isinstance(item, dict):
            missing_fields += 1
            continue
        evidence, reference_count, invalid_ids = _angle_evidence(item, summaries)
        evidence_references += reference_count
        invalid_evidence_ids += invalid_ids
        missing = [field for field in required if not str(item.get(field, "")).strip()]
        matched_evidence += len(evidence)
        if not str(item.get("title", "")).strip() or not str(
            item.get("central_argument", "")
        ).strip():
            missing_fields += 1
            continue
        if (strict and missing) or (strict and not evidence):
            missing_fields += 1
            continue
        title = str(item["title"]).strip()
        normalized_title = _normalized(title)
        if normalized_title in seen_titles:
            duplicate_titles += 1
            continue
        seen_titles.add(normalized_title)
        angles.append(
            NewsletterAngle(
                title=title,
                audience=str(item.get("audience", "")).strip(),
                central_argument=str(item.get("central_argument", "")).strip(),
                why_interesting=str(item.get("why_interesting", "")).strip(),
                rationale=str(item.get("rationale", "")).strip(),
                reporting_needed=str(item.get("reporting_needed", "")).strip(),
                evidence=evidence,
            )
        )
    stats = {
        "parsed": parsed_count,
        "missing_fields": missing_fields,
        "duplicate_titles": duplicate_titles,
        "evidence_references": evidence_references,
        "matched_evidence": matched_evidence,
        "invalid_evidence_ids": invalid_evidence_ids,
        "available_evidence": sum(len(summary.evidence or []) for summary in summaries),
        "accepted": len(angles),
    }
    LOGGER.info("Newsletter angle validation: %s", stats)
    if not angles and raise_on_failure:
        reasons = []
        if not summaries or not any(summary.evidence for summary in summaries):
            reasons.append("source evidence is insufficient")
        elif evidence_references == 0:
            reasons.append("the model returned no evidence references")
        elif invalid_evidence_ids:
            reasons.append("the model returned unknown evidence IDs")
        elif matched_evidence < evidence_references:
            reasons.append("some evidence IDs could not be validated")
        if duplicate_titles:
            reasons.append("the model generated duplicate angle titles")
        if missing_fields:
            reasons.append("one or more angles were malformed or missing required fields")
        detail = "; ".join(reasons) or "fewer than two valid angles were returned"
        raise BriefGenerationError(f"{ANGLE_ERROR_PREFIX} {detail}")
    return angles, stats


def _evidence_claims(summaries: list[SourceSummary]) -> list[EvidenceExcerpt]:
    return [
        evidence
        for summary in summaries
        for evidence in summary.evidence
        if evidence.category in {"verified_fact", "source_reported_claim"}
    ]


def _attributed_claim(evidence: EvidenceExcerpt) -> str:
    reference = f"{evidence.title} ({evidence.url})" if evidence.url else evidence.title
    return f"{evidence.claim} — {reference}"


def _fact_strings(summaries: list[SourceSummary]) -> list[str]:
    facts = []
    evidence_by_claim = {
        (item.source_id, _normalized(item.claim)): item
        for item in summaries
        for item in item.evidence
        if item.category in {"verified_fact", "source_reported_claim"}
    }
    for summary in summaries:
        for fact in summary.verified_facts:
            evidence = evidence_by_claim.get((summary.source_id, _normalized(fact)))
            label = (
                "Directly stated in source"
                if evidence and evidence.category == "verified_fact"
                else (
                    "Source-attributed claim (excerpt not matched)"
                    if evidence is None
                    else "Source-reported claim"
                )
            )
            facts.append(f"[{label}] {_attributed_claim(evidence) if evidence else f'{fact} — {_summary_reference(summary)}'}")
    for evidence in _evidence_claims(summaries):
        label = (
            "Directly stated in source"
            if evidence.category == "verified_fact"
            else "Source-reported claim"
        )
        facts.append(f"[{label}] {_attributed_claim(evidence)}")
    return _unique(facts)


def _source_findings_text(summaries: list[SourceSummary]) -> str:
    parts = []
    for summary in summaries:
        claims = summary.verified_facts[:3] or [
            evidence.claim for evidence in summary.evidence[:3]
        ]
        if claims:
            parts.append(
                f"{_summary_reference(summary)} reports: {' '.join(claims)}"
            )
    return "Source-by-source findings (not cross-source synthesis): " + " ".join(parts)


def _parse_brief(
    payload: Any,
    summaries: list[SourceSummary],
) -> ResearchBrief:
    if not isinstance(payload, dict):
        raise BriefGenerationError("The model returned an invalid brief format.")

    evidence = _unique(
        [_replace_source_ids(item, summaries) for item in _as_string_list(payload.get("evidence"))]
    )
    known_source_links = [summary.url for summary in summaries if summary.url]
    evidence.extend(link for link in known_source_links if link not in evidence)
    evidence_records = []
    known = {
        (item.source_id, item.excerpt)
        for summary in summaries
        for item in (summary.evidence or [])
    }
    for item in payload.get("evidence_records", []):
        if not isinstance(item, dict):
            continue
        key = (str(item.get("source_id", "")), str(item.get("excerpt", "")))
        if key not in known:
            continue
        matched = next(
            evidence for summary in summaries for evidence in (summary.evidence or [])
            if (evidence.source_id, evidence.excerpt) == key
        )
        evidence_records.append(matched)
    verified_facts = _fact_strings(summaries)
    analysis_value = (
        payload.get("analysis")
        if payload.get("analysis") is not None
        else payload.get("editorial_insight")
    )
    analysis_items = (
        [analysis_value]
        if isinstance(analysis_value, str)
        else _as_string_list(analysis_value)
    )
    analysis = _unique([
        _replace_source_ids(item, summaries) for item in analysis_items
    ])
    speculation = _unique([
        _replace_source_ids(item, summaries) for item in _as_string_list(payload.get("speculation"))
        if item not in verified_facts and item not in analysis
    ])
    open_questions = _unique([
        _replace_source_ids(item, summaries) for item in _as_string_list(payload.get("open_questions"))
        if item not in verified_facts and item not in analysis and item not in speculation
    ])
    if not verified_facts:
        raise BriefGenerationError(
            "No qualifying source facts or evidence excerpts were available for the brief."
        )
    implications = _unique([
        _replace_source_ids(item, summaries)
        for item in _as_string_list(payload.get("implications"))
    ])
    comparison_value = (
        payload.get("source_comparison")
        if payload.get("source_comparison") is not None
        else payload.get("source_relationship")
    )
    comparison_items = (
        [comparison_value]
        if isinstance(comparison_value, str)
        else _as_string_list(comparison_value)
    )
    source_comparison = [
        _replace_source_ids(item, summaries)
        for item in comparison_items
    ]
    overall_finding = payload.get("overall_finding")
    what_happened = _replace_source_ids(
        str(
            payload.get("what_happened")
            if payload.get("what_happened") is not None
            else overall_finding or ""
        ).strip(),
        summaries,
    )
    if payload.get("implications") is None:
        implications = _unique(
            _replace_source_ids(item, summaries)
            for item in _as_string_list(payload.get("practical_implications"))
        )
    why_it_matters = _replace_source_ids(
        str(payload.get("why_it_matters", "")).strip(), summaries
    )
    compact_synthesis = bool(payload.get("_compact_synthesis")) or (
        "overall_finding" in payload
        and "practical_implications" in payload
        and "executive_summary" not in payload
    )
    practical_implications = implications
    if compact_synthesis and not why_it_matters:
        why_it_matters = " ".join(practical_implications)
    who_should_care = [
        _replace_source_ids(item, summaries)
        for item in _as_string_list(payload.get("who_should_care"))
    ]
    if compact_synthesis and not who_should_care:
        audience_labels = (
            ("founders", "Founders"),
            ("builders", "Builders"),
            ("business operators", "Business operators"),
        )
        who_should_care = [
            label
            for term, label in audience_labels
            if any(term in item.lower() for item in practical_implications)
        ]
    editorial_insight = payload.get("editorial_insight")
    most_significant_finding = _replace_source_ids(
        str(
            payload.get("most_significant_finding")
            if payload.get("most_significant_finding") is not None
            else editorial_insight or ""
        ).strip(),
        summaries,
    )
    if not evidence_records:
        evidence_records = _evidence_claims(summaries)
    executive_summary = _replace_source_ids(
        str(
            payload.get("executive_summary")
            if payload.get("executive_summary") is not None
            else overall_finding or ""
        ).strip(),
        summaries,
    )
    missing_sections = [
        name
        for name, value in (
            ("executive_summary", executive_summary),
            ("what_happened", what_happened),
            ("most_significant_finding", most_significant_finding),
            ("why_it_matters", why_it_matters),
            ("who_should_care", who_should_care),
            ("source_comparison", source_comparison),
            ("analysis", analysis),
            ("implications", implications),
        )
        if not value
    ]
    if (
        most_significant_finding
        and what_happened
        and _normalized(most_significant_finding) == _normalized(what_happened)
    ):
        distinct_signal = next(
            (
                item
                for item in analysis + implications + open_questions
                if _normalized(item) not in {
                    _normalized(what_happened),
                    _normalized(executive_summary),
                }
            ),
            "",
        )
        if distinct_signal:
            most_significant_finding = distinct_signal
    return ResearchBrief(
        executive_summary=executive_summary,
        what_happened=what_happened,
        why_it_matters=why_it_matters,
        who_should_care=who_should_care,
        verified_facts=verified_facts,
        analysis=analysis,
        speculation=speculation,
        evidence=evidence,
        newsletter_angles=[],
        open_questions=open_questions,
        source_comparison=source_comparison,
        most_significant_finding=most_significant_finding,
        implications=implications,
        evidence_records=evidence_records,
        angle_details=[],
        recommended_angle="",
        source_findings=_source_findings(summaries),
        newsletter_angle_limitation="",
        missing_sections=missing_sections,
    )


def _missing_synthesis_fields(payload: dict[str, Any]) -> list[str]:
    if payload.get("_compact_synthesis") is False:
        return [
            field
            for field in (
                "executive_summary",
                "what_happened",
                "most_significant_finding",
                "source_comparison",
                "analysis",
                "implications",
            )
            if not payload.get(field)
        ]

    def meaningful(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value.strip()) >= 12
            and value.strip().lower() not in {"...", "n/a", "unknown", "none"}
        )

    missing: list[str] = []
    for field in (
        "overall_finding",
        "source_relationship",
        "editorial_insight",
    ):
        if not meaningful(payload.get(field)):
            missing.append(field)
    implications = payload.get("practical_implications")
    if (
        not isinstance(implications, list)
        or not implications
        or any(not meaningful(item) for item in implications)
    ):
        missing.append("practical_implications")
    elif payload.get("_compact_synthesis", True) and not any(
        any(
            audience in item.lower()
            for audience in ("founders", "builders", "business operators")
        )
        for item in implications
    ):
        missing.append("practical_implications (must identify an audience)")
    return missing


def _normalize_synthesis_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep callers that still provide the retired synthesis shape compatible."""
    if any(
        field in payload
        for field in (
            "overall_finding",
            "source_relationship",
            "editorial_insight",
            "practical_implications",
        )
    ):
        return {**payload, "_compact_synthesis": True}
    return {
        "overall_finding": payload.get("executive_summary", ""),
        "source_relationship": " ".join(
            _as_string_list(payload.get("source_comparison"))
        ),
        "editorial_insight": payload.get("most_significant_finding", ""),
        "practical_implications": _as_string_list(payload.get("implications")),
        "_compact_synthesis": False,
        **payload,
    }


def _synthesis_retry_prompt(summaries: list[SourceSummary]) -> str:
    return f"""Return only this JSON object:
{{"overall_finding":"...","source_relationship":"...","editorial_insight":"...",
"practical_implications":["...","..."]}}
Use only the supplied evidence. State what the sources collectively establish, how they
agree or differ, and identify the most important cross-source insight.
Summarize what each source reports and provide exactly two practical implications. Each
implication must name who should care using one or more of: founders, builders, business
operators. Format each as "Audience: implication". This is also the evidence-grounded
answer to why the development matters.
Do not invent facts, metadata, evidence, or causality.

SOURCE CONTEXT:
{_source_context(summaries)}
"""


def _angle_prompt(summaries: list[SourceSummary], diagnostic: str = "") -> str:
    retry_note = f"\nPrevious validation result: {diagnostic}\n" if diagnostic else ""
    return f"""Generate one primary newsletter angle and, if the evidence supports it,
up to two optional alternative angles. Use only the evidence catalogue below.
Each angle must contain non-empty title, audience, central_argument, why_interesting,
rationale, reporting_needed, and supporting_evidence_ids. supporting_evidence_ids must
contain only IDs from the catalogue. Do not return excerpts, URLs, or complete evidence
objects. A single valid angle is acceptable; never fabricate a second angle.
Return only JSON with this shape:
{{"angle_details": [{{"title": "...", "audience": "...",
"central_argument": "...", "why_interesting": "...", "rationale": "...",
"reporting_needed": "...", "supporting_evidence_ids": ["..."]}}]}}
{retry_note}
AVAILABLE EVIDENCE:
{json.dumps(_available_evidence(summaries), ensure_ascii=False)}
"""


def _angle_limitation(stats: dict[str, int]) -> str:
    if stats["available_evidence"] == 0:
        return "Newsletter angles could not be generated because no validated source evidence was available."
    if stats["invalid_evidence_ids"]:
        return "Newsletter angles were limited because the model returned unknown evidence IDs."
    if stats["evidence_references"] == 0:
        return "Newsletter angles could not be generated because the model returned no supporting evidence IDs."
    if stats["missing_fields"]:
        return "Newsletter angles were limited because one or more model angle objects were missing required fields."
    return "No valid evidence-grounded newsletter angle was returned by the model."


def _generate_angles(summaries: list[SourceSummary]) -> tuple[list[NewsletterAngle], str]:
    """Generate optional angles independently from the main research synthesis."""
    messages = [
        {
            "role": "system",
            "content": "You generate concise, evidence-grounded editorial angles.",
        },
        {"role": "user", "content": _angle_prompt(summaries)},
    ]
    diagnostic = ""
    for attempt in range(2):
        try:
            content = chat_json(
                messages,
                timeout=LLM_TIMEOUT_SECONDS,
                max_tokens=1200,
                temperature=0.2,
            )
            LOGGER.debug(
                "Raw angle model response (truncated): %s",
                content[:4000] if isinstance(content, str) else repr(content)[:4000],
            )
            payload = _parse_json_object(content)
            angles, stats = _parse_angles(
                payload,
                summaries,
                strict=True,
                raise_on_failure=False,
            )
            if angles:
                limitation = (
                    ""
                    if len(angles) > 1
                    else "Only one valid evidence-grounded newsletter angle was produced."
                )
                return angles, limitation
            diagnostic = _angle_limitation(stats)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            diagnostic = f"The model returned malformed angle JSON ({type(exc).__name__})."
        except OllamaError as exc:
            diagnostic = f"The angle request failed: {exc}"
        except Exception as exc:
            diagnostic = f"Unexpected angle-generation error ({type(exc).__name__})."
        LOGGER.warning("Newsletter angle generation attempt %d failed: %s", attempt + 1, diagnostic)
        if attempt == 0:
            messages = [
                {
                    "role": "system",
                    "content": "Return only the requested JSON angle schema.",
                },
                {
                    "role": "user",
                    "content": _angle_prompt(summaries, diagnostic),
                },
            ]
    return [], diagnostic or "No valid evidence-grounded newsletter angle was returned by the model."


def generate_research_brief(summaries: list[SourceSummary]) -> ResearchBrief:
    prompt = f"""Return only this JSON object:
{{"overall_finding":"...","source_relationship":"...","editorial_insight":"...",
"practical_implications":["...","..."]}}

Use only the supplied source context and exact evidence excerpts. Do not invent facts,
metadata, evidence, or causality. Do not regenerate source summaries, evidence records,
source metadata, or newsletter angles. Keep each field concise:
- overall_finding: what the sources collectively establish
- source_relationship: how they agree, differ, or complement one another
- editorial_insight: the most important insight emerging across sources
- practical_implications: two evidence-supported implications, each naming who should
  care (founders, builders, or business operators) in the form "Audience: implication"

SOURCE CONTEXT:
{_source_context(summaries)}
"""

    messages = [
        {
            "role": "system",
            "content": "You produce cautious, source-grounded editorial research briefs.",
        },
        {"role": "user", "content": prompt},
    ]
    for attempt in range(2):
        try:
            content = chat_json(
                messages,
                timeout=LLM_TIMEOUT_SECONDS,
                max_tokens=2600,
                temperature=0.2,
            )
            LOGGER.debug(
                "Raw synthesis model response (truncated): %s",
                content[:6000] if isinstance(content, str) else repr(content)[:6000],
            )
            parsed_payload = _normalize_synthesis_payload(_parse_json_object(content))
            missing_fields = _missing_synthesis_fields(parsed_payload)
            LOGGER.info(
                "Parsed synthesis response: keys=%s missing_required=%s",
                sorted(parsed_payload.keys()),
                missing_fields,
            )
            if missing_fields:
                message = (
                    "Synthesis response was missing required editorial fields: "
                    + ", ".join(missing_fields)
                )
                LOGGER.warning("%s", message)
                if attempt == 0:
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "Return only the requested four-field JSON object."
                            ),
                        },
                        {
                            "role": "user",
                            "content": _synthesis_retry_prompt(summaries),
                        },
                    ]
                    continue
                raise BriefGenerationError(
                    message
                    + ". The retry also returned an incomplete synthesis."
                )
            brief = _parse_brief(parsed_payload, summaries)
            angles, limitation = _generate_angles(summaries)
            return replace(
                brief,
                angle_details=angles,
                recommended_angle=angles[0].title if angles else "",
                newsletter_angle_limitation=limitation,
            )
        except BriefGenerationError:
            raise
        except OllamaError as exc:
            raise BriefGenerationError(str(exc)) from exc
        except json.JSONDecodeError as exc:
            if attempt == 0:
                messages = [
                    {
                        "role": "system",
                        "content": "Return only the requested four-field JSON object.",
                    },
                    {
                        "role": "user",
                        "content": _synthesis_retry_prompt(summaries),
                    },
                ]
                continue
            raise BriefGenerationError(
                "Qwen returned malformed or truncated JSON for the research brief. "
                "Retry with fewer sources or shorter source text."
            ) from exc
        except (KeyError, IndexError, TypeError) as exc:
            raise BriefGenerationError("The model returned an unreadable brief.") from exc
        except Exception as exc:
            raise BriefGenerationError(
                f"Unexpected AI error ({type(exc).__name__}): {exc}"
            ) from exc
    raise BriefGenerationError("The model did not return a research brief.")
