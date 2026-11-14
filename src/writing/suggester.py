"""Suggestion and rewrite generation."""

from src.models.schemas import WritingSuggestion


def apply_suggestions(
    original_text: str, suggestions: list[WritingSuggestion], accepted_indices: set[int]
) -> str:
    revised_text = original_text
    for idx, suggestion in enumerate(suggestions):
        if idx not in accepted_indices:
            continue
        if suggestion.original in revised_text:
            revised_text = revised_text.replace(
                suggestion.original,
                suggestion.suggestion,
                1,
            )
    return revised_text
