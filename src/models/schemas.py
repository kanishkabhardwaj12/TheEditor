"""Data schemas for research and writing workflows."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Source:
    raw_input: str
    url: Optional[str] = None
    title: str = ""
    content: str = ""
    source_type: str = "unknown"
    status: str = "pending"
    error: Optional[str] = None
    extraction_method: str = ""
    http_status: Optional[int] = None
    word_count: int = 0


@dataclass
class SourceSummary:
    source_id: str
    title: str
    url: Optional[str]
    verified_facts: list[str]
    important_context: list[str]
    caveats: list[str]
    claims_to_verify: list[str]
    evidence: list["EvidenceExcerpt"] = field(default_factory=list)
    topic: str = ""
    central_argument: str = ""
    methodology: str = ""
    sample_size: str = ""
    population: str = ""
    date: str = ""
    implications: list[str] = field(default_factory=list)
    findings: list["SourceFinding"] = field(default_factory=list)


@dataclass
class EvidenceExcerpt:
    source_id: str
    title: str
    url: Optional[str]
    claim: str
    excerpt: str
    category: str = "verified_fact"
    evidence_id: str = ""


@dataclass
class SourceFinding:
    source_id: str
    title: str
    url: Optional[str]
    claim: str
    evidence: list[EvidenceExcerpt] = field(default_factory=list)
    methodology: str = ""
    sample_size: str = ""
    population: str = ""
    date: str = ""
    limitation: str = ""
    implication: str = ""


@dataclass
class NewsletterAngle:
    title: str
    audience: str
    central_argument: str
    why_interesting: str
    rationale: str = ""
    evidence: list[EvidenceExcerpt] = field(default_factory=list)
    reporting_needed: str = ""


@dataclass
class WritingSuggestion:
    issue_type: str
    original: str
    explanation: str
    suggestion: str
    start: int = 0
    end: int = 0
    severity: str = "medium"
    confidence: float = 0.0
    source: str = "deterministic"
    can_apply: bool = True
    suggestion_id: str = ""


@dataclass
class ResearchBrief:
    executive_summary: str
    what_happened: str
    why_it_matters: str
    who_should_care: list[str]
    verified_facts: list[str]
    analysis: list[str]
    speculation: list[str]
    evidence: list[str]
    newsletter_angles: list[str]
    open_questions: list[str]
    source_comparison: list[str] = field(default_factory=list)
    most_significant_finding: str = ""
    implications: list[str] = field(default_factory=list)
    evidence_records: list[EvidenceExcerpt] = field(default_factory=list)
    angle_details: list[NewsletterAngle] = field(default_factory=list)
    recommended_angle: str = ""
    source_findings: list[SourceFinding] = field(default_factory=list)
    newsletter_angle_limitation: str = ""
    missing_sections: list[str] = field(default_factory=list)
