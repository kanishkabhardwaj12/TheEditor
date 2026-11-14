"""Layered URL extraction with explicit status and extraction telemetry."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple
from urllib.parse import urljoin

import httpx
import trafilatura
from bs4 import BeautifulSoup

from src.models.schemas import Source

LOGGER = logging.getLogger(__name__)
REQUEST_TIMEOUT_SECONDS = 12
MIN_CONTENT_CHARACTERS = 120
MIN_PARTIAL_CHARACTERS = 60
WORD_COUNT_MINIMUM = 25
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/rss+xml,application/atom+xml",
    "Accept-Language": "en-US,en;q=0.8",
}


def _word_count(content: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", content))


def _usable(content: str, minimum: int = MIN_CONTENT_CHARACTERS) -> bool:
    return len(content.strip()) >= minimum and _word_count(content) >= WORD_COUNT_MINIMUM


def _title_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.title.get_text(" ", strip=True) if soup.title else ""


def _extract_with_trafilatura(html: str) -> Tuple[str, str]:
    content = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    ) or ""
    return _title_from_html(html), content.strip()


def _extract_with_beautifulsoup(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        element.decompose()
    container = soup.find("article") or soup.find("main") or soup.body
    if container is None:
        return title, ""
    paragraphs = [
        paragraph.get_text(" ", strip=True)
        for paragraph in container.find_all("p")
    ]
    content = "\n\n".join(paragraph for paragraph in paragraphs if paragraph)
    return title, content.strip()


def _extract_rss(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "xml")
    entries = soup.find_all(["item", "entry"])
    if not entries:
        return "", ""
    title = entries[0].find("title")
    parts = []
    for entry in entries[:10]:
        for tag_name in ("title", "description", "summary", "content"):
            tag = entry.find(tag_name)
            if tag:
                value = tag.get_text(" ", strip=True)
                if value:
                    parts.append(value)
    return (title.get_text(" ", strip=True) if title else "", "\n\n".join(parts))


def _alternate_feed_url(html: str, page_url: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    link = soup.find(
        "link",
        rel=lambda value: value and "alternate" in value,
        type=lambda value: value in {"application/rss+xml", "application/atom+xml"},
    )
    href = link.get("href") if link else None
    return urljoin(page_url, href) if href else None


def _blocked_status(status_code: int) -> bool:
    return status_code in {401, 403, 407, 429, 451}


def _set_result(
    source: Source,
    *,
    title: str,
    content: str,
    status: str,
    method: str,
    error: Optional[str] = None,
) -> Source:
    source.title = title or source.title or source.url or "Untitled source"
    source.content = content
    source.status = status
    source.extraction_method = method
    source.word_count = _word_count(content)
    source.error = error
    LOGGER.info(
        "URL extraction url=%s method=%s http_status=%s status=%s words=%d reason=%s",
        source.url,
        method or "none",
        source.http_status or "none",
        status,
        source.word_count,
        error or "none",
    )
    return source


def extract_source(source: Source) -> Source:
    if source.source_type != "url" or not source.url:
        return source

    try:
        with httpx.Client(
            headers=HEADERS,
            follow_redirects=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as client:
            response = client.get(source.url)
    except httpx.TimeoutException:
        return _set_result(
            source,
            title=source.url,
            content="",
            status="failed",
            method="httpx",
            error="The request timed out. Paste the source text manually.",
        )
    except httpx.RequestError as exc:
        return _set_result(
            source,
            title=source.url,
            content="",
            status="failed",
            method="httpx",
            error=f"Could not fetch this URL: {exc}",
        )

    source.http_status = response.status_code
    if _blocked_status(response.status_code):
        return _set_result(
            source,
            title=source.url,
            content="",
            status="blocked",
            method="httpx",
            error=(
                f"The publisher blocked automated access (HTTP {response.status_code}). "
                "Open the link in a browser and paste the article text manually."
            ),
        )
    if response.status_code >= 400:
        return _set_result(
            source,
            title=source.url,
            content="",
            status="failed",
            method="httpx",
            error=f"The publisher returned HTTP {response.status_code}.",
        )

    html = response.text
    title, content = _extract_with_trafilatura(html)
    if _usable(content):
        return _set_result(
            source, title=title, content=content, status="extracted", method="trafilatura"
        )

    fallback_title, fallback_content = _extract_with_beautifulsoup(html)
    if _usable(fallback_content):
        return _set_result(
            source,
            title=fallback_title or title,
            content=fallback_content,
            status="extracted",
            method="httpx+beautifulsoup",
        )
    if len(fallback_content.strip()) >= MIN_PARTIAL_CHARACTERS:
        return _set_result(
            source,
            title=fallback_title or title,
            content=fallback_content,
            status="partially_extracted",
            method="httpx+beautifulsoup",
            error="Only partial article content was readable; verify it before using it.",
        )

    feed_url = _alternate_feed_url(html, str(response.url))
    if feed_url and feed_url != source.url:
        try:
            feed_response = httpx.get(
                feed_url,
                headers=HEADERS,
                follow_redirects=True,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if feed_response.status_code < 400:
                rss_title, rss_content = _extract_rss(feed_response.text)
                if _usable(rss_content, MIN_PARTIAL_CHARACTERS):
                    source.http_status = feed_response.status_code
                    return _set_result(
                        source,
                        title=rss_title or title,
                        content=rss_content,
                        status="partially_extracted",
                        method="official rss/atom",
                        error="Used the publisher's official feed because article text was unavailable.",
                    )
        except httpx.HTTPError as exc:
            LOGGER.info("Official feed extraction failed url=%s reason=%s", feed_url, exc)

    rss_title, rss_content = _extract_rss(html)
    if _usable(rss_content, MIN_PARTIAL_CHARACTERS):
        return _set_result(
            source,
            title=rss_title or title,
            content=rss_content,
            status="partially_extracted",
            method="rss/atom",
            error="Used the page's feed content because article text was unavailable.",
        )

    js_hint = " The page may require JavaScript to render." if re.search(
        r"__NEXT_DATA__|id=[\"']root[\"']|enable javascript", html, re.I
    ) else ""
    return _set_result(
        source,
        title=title or source.url,
        content="",
        status="blocked",
        method="trafilatura+httpx+beautifulsoup",
        error=(
            "No relevant article text was available; automated extraction was not retried."
            + js_hint
            + " Paste the source content manually."
        ),
    )


def extract_sources(sources: list[Source]) -> list[Source]:
    if not sources:
        return []
    with ThreadPoolExecutor(max_workers=min(5, len(sources))) as executor:
        return list(executor.map(extract_source, sources))
