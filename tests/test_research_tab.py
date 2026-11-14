import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.ui.research_tab import (
    _items,
    _render_angle,
    _render_source_finding,
    _safe_link,
)


class ResearchTabRenderingTests(unittest.TestCase):
    def test_only_http_urls_become_clickable_links(self):
        self.assertIn('rel="noopener noreferrer"', _safe_link("https://example.com", "Source"))
        self.assertIn('rel="noopener noreferrer"', _safe_link("http://example.com", "Source"))
        self.assertNotIn("target=", _safe_link("javascript:alert(1)", "Unsafe"))
        self.assertNotIn("target=", _safe_link("file:///tmp/source", "Unsafe"))

    def test_link_attributes_and_untrusted_labels_are_escaped(self):
        rendered = _safe_link("https://example.com/?q=1&x=2", '<Source>')
        self.assertIn("https://example.com/?q=1&amp;x=2", rendered)
        self.assertIn("&lt;Source&gt;", rendered)

    @patch("src.ui.research_tab.st.markdown")
    def test_source_finding_tolerates_missing_optional_metadata(self, markdown):
        _render_source_finding(
            SimpleNamespace(
                title="Pasted source",
                source_id="source-1",
                url=None,
                claim="A useful finding.",
            )
        )
        rendered = markdown.call_args.args[0]
        self.assertIn("A useful finding.", rendered)
        self.assertNotIn("Method:", rendered)
        self.assertNotIn("Sample:", rendered)

    @patch("src.ui.research_tab.st.markdown")
    def test_angle_rendering_supports_sparse_angles(self, markdown):
        _render_angle(SimpleNamespace(title="One angle"))
        rendered = markdown.call_args.args[0]
        self.assertIn("One angle", rendered)
        self.assertIn("No detail available.", rendered)

    def test_empty_lists_have_explicit_empty_state(self):
        self.assertIn("No items available.", _items([]))


if __name__ == "__main__":
    unittest.main()
