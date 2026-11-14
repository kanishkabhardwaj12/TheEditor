"""Offset-safe editing helpers for the writing assistant."""

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Mapping, Tuple

from src.models.schemas import WritingSuggestion


@dataclass(frozen=True)
class AppliedEdit:
    suggestion_id: str
    start: int
    end: int
    original: str
    replacement: str


def coerce_applied_edit(edit: object) -> AppliedEdit:
    """Convert current and legacy session-state edits to the typed edit record."""
    if isinstance(edit, AppliedEdit):
        return edit
    if isinstance(edit, Mapping):
        original = str(edit.get("original", ""))
        replacement = str(edit.get("replacement", ""))
        start = int(edit.get("start", -1))
        end = int(edit.get("end", start + len(original)))
        suggestion_id = str(edit.get("suggestion_id", f"{start}:{end}:legacy"))
        return AppliedEdit(suggestion_id, start, end, original, replacement)
    raise TypeError(f"Unsupported applied edit type: {type(edit).__name__}")


def suggestion_key(suggestion: WritingSuggestion) -> str:
    if suggestion.suggestion_id:
        return suggestion.suggestion_id
    return f"{suggestion.start}:{suggestion.end}:{suggestion.issue_type}:{suggestion.original}"


def validate_suggestion(text: str, suggestion: WritingSuggestion) -> bool:
    if not (
        suggestion.start >= 0
        and suggestion.end > suggestion.start
        and suggestion.end <= len(text)
        and text[suggestion.start : suggestion.end] == suggestion.original
        and suggestion.suggestion.strip() != suggestion.original.strip()
    ):
        return False
    original_words = {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z'-]+", suggestion.original)
    }
    replacement_words = {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z'-]+", suggestion.suggestion)
    }
    if (
        original_words
        and replacement_words
        and len(original_words & replacement_words) / len(original_words) < 0.35
    ):
        return False
    if suggestion.original.strip().endswith((".", "!", "?")):
        replacement = suggestion.suggestion.strip()
        if not replacement.endswith((".", "!", "?")):
            return False
        if not re.search(
            r"\b(?:am|are|is|was|were|be|been|being|"
            r"wrote|written|made|led|found|came|went|did|had|has|have|"
            r"can|could|will|would|should|[A-Za-z]+ed|[A-Za-z]+ing)\b",
            replacement,
            re.I,
        ):
            return False
    return True


def non_overlapping(suggestions: Iterable[WritingSuggestion]) -> List[WritingSuggestion]:
    selected: List[WritingSuggestion] = []
    for suggestion in sorted(
        suggestions,
        key=lambda item: (
            item.issue_type,
            item.start,
            -(item.end - item.start),
            item.source != "deterministic",
        ),
    ):
        if any(
            existing.issue_type == suggestion.issue_type
            and suggestion.start < existing.end
            and suggestion.end > existing.start
            for existing in selected
        ):
            continue
        selected.append(suggestion)
    return selected


def build_current_draft(
    original: str, edits: Iterable[AppliedEdit]
) -> Tuple[str, Dict[str, Tuple[int, int]]]:
    """Apply original-coordinate edits and return current text plus replacement ranges."""
    normalized: List[AppliedEdit] = []
    for edit in (coerce_applied_edit(item) for item in edits):
        if (
            edit.start < 0
            or edit.end > len(original)
            or original[edit.start : edit.end] != edit.original
        ):
            legacy_start = original.find(edit.original)
            if legacy_start < 0:
                continue
            edit = AppliedEdit(
                edit.suggestion_id,
                legacy_start,
                legacy_start + len(edit.original),
                edit.original,
                edit.replacement,
            )
        normalized.append(edit)
    ordered = sorted(normalized, key=lambda edit: edit.start)
    cursor = 0
    parts: List[str] = []
    ranges: Dict[str, Tuple[int, int]] = {}
    current_length = 0
    for edit in ordered:
        if edit.start < cursor or original[edit.start : edit.end] != edit.original:
            continue
        unchanged = original[cursor : edit.start]
        parts.append(unchanged)
        current_length += len(unchanged)
        replacement_start = current_length
        parts.append(edit.replacement)
        current_length += len(edit.replacement)
        ranges[edit.suggestion_id] = (replacement_start, current_length)
        cursor = edit.end
    parts.append(original[cursor:])
    return "".join(parts), ranges
