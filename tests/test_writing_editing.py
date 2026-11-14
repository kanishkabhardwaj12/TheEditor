import unittest

from src.models.schemas import WritingSuggestion
from src.writing.editing import (
    AppliedEdit,
    build_current_draft,
    non_overlapping,
    validate_suggestion,
)


class WritingEditingTests(unittest.TestCase):
    def test_multiple_original_coordinate_edits_preserve_offsets(self):
        original = "Alpha sentence. Beta sentence. Gamma sentence."
        first = AppliedEdit("first", 0, 15, "Alpha sentence.", "Alpha revised.")
        second = AppliedEdit("second", 16, 30, "Beta sentence.", "Beta revised.")

        current, ranges = build_current_draft(original, [first, second])

        self.assertEqual(current, "Alpha revised. Beta revised. Gamma sentence.")
        self.assertEqual(current[ranges["first"][0] : ranges["first"][1]], "Alpha revised.")
        self.assertEqual(current[ranges["second"][0] : ranges["second"][1]], "Beta revised.")

    def test_stale_suggestion_is_rejected(self):
        suggestion = WritingSuggestion(
            "Clarity",
            "Alpha sentence.",
            "Use a more direct sentence.",
            "Alpha revised.",
            0,
            15,
        )
        self.assertFalse(validate_suggestion("Changed sentence.", suggestion))

    def test_overlapping_suggestions_keep_one(self):
        first = WritingSuggestion("A", "Alpha sentence.", "x", "Alpha revised.", 0, 15)
        overlap = WritingSuggestion(
            "B", "Alpha sentence. Beta", "x", "Revised text.", 0, 21
        )
        selected = non_overlapping([first, overlap])
        self.assertEqual(selected, [first, overlap])

    def test_incomplete_rewrite_is_rejected(self):
        suggestion = WritingSuggestion(
            "Clarity",
            "The original sentence.",
            "Incomplete rewrite.",
            "And then the",
            0,
            22,
        )
        self.assertFalse(validate_suggestion("The original sentence.", suggestion))


if __name__ == "__main__":
    unittest.main()
