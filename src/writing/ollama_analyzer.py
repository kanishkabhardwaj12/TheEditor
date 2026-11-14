"""Local Qwen analysis for jargon and paragraph clarity."""

import json
from typing import Any, List

from src.llm.ollama import OllamaError, chat_json, get_model
from src.models.schemas import WritingSuggestion
from src.writing.editing import validate_suggestion

class OllamaAnalysisError(RuntimeError):
    """Raised when the local Ollama analysis cannot be completed."""


def _list(value: Any) -> List[str]:
    return [item for item in value if isinstance(item, str) and item.strip()] if isinstance(value, list) else []


def _analyze_category(text: str, category: str, timeout: int) -> List[dict]:
    prompt = f"""You are an editorial writing assistant.

    Analyze the draft for meaningful instances of {category}.
    Return no more than 2 genuine issues. Return an empty issues array if there are none.

    For Jargon, flag only difficult technical terms. Never flag company names, publication names, source titles, URLs, or Markdown links.

    For Paragraph clarity, flag only genuinely confusing passages.

    For Semantic repetition, flag only passages that repeat an idea already expressed elsewhere. Do not rewrite passages merely to improve their wording.

    For every issue, return:
    - issue_type
    - original
    - start
    - end
    - explanation
    - suggestion
    - severity
    - confidence

    Rules:
    - original must be copied EXACTLY from the draft, including punctuation and Markdown.
    - start is the zero-based character position where original begins in the draft.
    - end is start plus the length of original.
    - suggestion must replace original, not rewrite the entire article.
    - Preserve all factual claims and Markdown links.
    - Do not return unchanged suggestions.
    - severity must be "low", "medium", or "high".
    - confidence must be between 0 and 1.
    - Do not use alternative field names such as "replacement" or "replacement_text".

    Return ONLY JSON in this format:
    {{
    "issues": [
        {{
        "issue_type": "{category}",
        "original": "Exact passage from draft",
        "start": 0,
        "end": 20,
        "explanation": "Why this is a genuine issue",
        "suggestion": "Replacement passage",
        "severity": "medium",
        "confidence": 0.85
        }}
    ]
    }}

    If there are no issues, return:
    {{"issues": []}}

    DRAFT:
    {text}
    """
    try:
        content = chat_json(
            [{"role": "user", "content": prompt}],
            timeout=timeout,
            max_tokens=2000,
            temperature=0,
        )
        print(f"\n--- QWEN {category} RESPONSE ---\n{content}\n")
        payload = json.loads(content)
        issues = payload.get("issues", [])
        if not isinstance(issues, list):
            raise ValueError("issues is not an array")
    except (OllamaError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise OllamaAnalysisError(
            f"Local Qwen analysis failed. Make sure Ollama is running and {get_model()} is available."
        ) from exc

    return [issue for issue in issues if isinstance(issue, dict)]


def analyze_with_ollama(text: str, timeout: int = 90) -> List[WritingSuggestion]:
    issues: List[dict] = []
    for category in ("Jargon", "Paragraph clarity", "Semantic repetition"):
        issues.extend(_analyze_category(text, category, timeout))

    suggestions: List[WritingSuggestion] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        original = issue.get("original")
        suggestion = issue.get("suggestion")
        if not isinstance(original, str) or not isinstance(suggestion, str) or not original or not suggestion:
            continue
        try:
            proposed_start = int(issue.get("start", -1))
            proposed_end = int(issue.get("end", -1))
        except (TypeError, ValueError):
            proposed_start, proposed_end = -1, -1
        if (
            proposed_start >= 0
            and proposed_end == proposed_start + len(original)
            and text[proposed_start:proposed_end] == original
        ):
            start = proposed_start
        else:
            start = text.find(original)
        if start < 0:
            continue
        issue_type = str(issue.get("issue_type", "Jargon"))
        if issue_type not in {"Jargon", "Paragraph clarity", "Semantic repetition"}:
            continue
        if not validate_suggestion(
            text,
            WritingSuggestion(
                issue_type,
                original,
                str(issue.get("explanation", "")).strip(),
                suggestion,
                start,
                start + len(original),
            ),
        ):
            continue
        try:
            confidence = float(issue.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7
        suggestions.append(
            WritingSuggestion(
                issue_type=issue_type,
                original=original,
                explanation=str(issue.get("explanation", "")).strip(),
                suggestion=suggestion,
                start=start,
                end=start + len(original),
                severity=str(issue.get("severity", "medium")),
                confidence=max(0.0, min(1.0, confidence)),
                source="ollama",
                suggestion_id=f"{start}:{start + len(original)}:{issue_type}",
            )
        )
    return suggestions
