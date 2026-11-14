import unittest

from src.writing.analyzer import analyze_draft
from src.writing.editing import non_overlapping
from src.models.schemas import WritingSuggestion


class WritingAnalysisTests(unittest.TestCase):
    def test_independent_detectors_can_report_same_sentence(self):
        text = "The report was basically written by the team."
        findings = analyze_draft(text)
        issue_types = {item.issue_type for item in findings}
        self.assertIn("Passive voice", issue_types)
        self.assertIn("Filler phrase", issue_types)

    def test_passive_rewrites_are_clause_scoped(self):
        text = "The report was written by the team, and it was reviewed by the editor."
        findings = [
            item for item in analyze_draft(text) if item.issue_type == "Passive voice"
        ]
        self.assertEqual(len(findings), 2)
        self.assertEqual(
            [text[item.start : item.end] for item in findings],
            ["The report was written by the team", "it was reviewed by the editor"],
        )

    def test_same_category_overlap_is_deduplicated_but_categories_remain(self):
        first = WritingSuggestion("Jargon", "technical term", "x", "plain term", 0, 14)
        duplicate = WritingSuggestion("Jargon", "technical term", "y", "simple term", 0, 14)
        other = WritingSuggestion("Paragraph clarity", "technical term", "z", "clear term", 0, 14)
        selected = non_overlapping([first, duplicate, other])
        self.assertEqual(len(selected), 2)
        self.assertEqual({item.issue_type for item in selected}, {"Jargon", "Paragraph clarity"})


if __name__ == "__main__":
    unittest.main()
