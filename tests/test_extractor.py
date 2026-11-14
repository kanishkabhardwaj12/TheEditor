import unittest
from unittest.mock import Mock, patch

from src.models.schemas import Source
from src.research.extractor import extract_source, extract_sources


ARTICLE_HTML = """
<html><head><title>Example article</title></head>
<body><article>
<p>This is a sufficiently long article paragraph with meaningful editorial content.</p>
<p>It contains a second paragraph so the extractor can distinguish article text from navigation.</p>
</article></body></html>
"""


class ExtractorTests(unittest.TestCase):
    def _response(self, status_code=200, text=ARTICLE_HTML):
        response = Mock()
        response.status_code = status_code
        response.text = text
        response.url = "https://example.com/article"
        response.raise_for_status.return_value = None
        return response

    @patch("src.research.extractor.httpx.Client")
    def test_successful_extraction_uses_trafilatura_or_fallback(self, client_class):
        client = client_class.return_value.__enter__.return_value
        client.get.return_value = self._response()
        source = extract_source(
            Source("https://example.com/article", "https://example.com/article", source_type="url")
        )
        self.assertIn(source.status, {"extracted", "partially_extracted"})
        self.assertGreater(source.word_count, 0)
        self.assertTrue(source.extraction_method)

    @patch("src.research.extractor.httpx.Client")
    def test_blocked_response_is_not_retried(self, client_class):
        client = client_class.return_value.__enter__.return_value
        client.get.return_value = self._response(403, "")
        source = extract_source(
            Source("https://example.com/article", "https://example.com/article", source_type="url")
        )
        self.assertEqual(source.status, "blocked")
        client.get.assert_called_once()

    @patch("src.research.extractor.httpx.Client")
    def test_empty_page_is_blocked_without_sending_content_downstream(self, client_class):
        client = client_class.return_value.__enter__.return_value
        client.get.return_value = self._response(
            200,
            "<html><body><nav>Home Login Subscribe</nav><div id='app'></div></body></html>",
        )
        source = extract_source(
            Source("https://example.com/article", "https://example.com/article", source_type="url")
        )
        self.assertEqual(source.status, "blocked")
        self.assertEqual(source.content, "")
        self.assertEqual(source.word_count, 0)

    @patch("src.research.extractor.extract_source")
    def test_failed_source_does_not_discard_successful_sources(self, extract):
        extract.side_effect = [
            Source("blocked", status="blocked", error="403"),
            Source(
                "usable",
                title="Usable source",
                content="A usable source with evidence.",
                status="extracted",
                word_count=6,
            ),
        ]
        results = extract_sources([Source("blocked"), Source("usable")])
        self.assertEqual(results[0].status, "blocked")
        self.assertEqual(results[1].title, "Usable source")


if __name__ == "__main__":
    unittest.main()
