"""Input ingestion utilities for URLs and pasted text."""

from urllib.parse import urlparse

from src.models.schemas import Source


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def parse_sources(urls_text: str, pasted_text: str) -> list[Source]:
    sources: list[Source] = []
    for raw in urls_text.splitlines():
        item = raw.strip()
        if item:
            if _is_http_url(item):
                sources.append(Source(raw_input=item, url=item, source_type="url"))
            else:
                sources.append(
                    Source(
                        raw_input=item,
                        source_type="url",
                        status="failed",
                        error="Enter a complete HTTP or HTTPS URL.",
                    )
                )

    if pasted_text.strip():
        chunks = [
            c.strip()
            for c in pasted_text.split("\n---SOURCE---\n")
            if c.strip()
        ]
        for index, chunk in enumerate(chunks, start=1):
            lines = [line.strip() for line in chunk.splitlines() if line.strip()]
            candidate_title = lines[0] if lines and len(lines[0]) <= 140 else ""
            sources.append(
                Source(
                    raw_input=chunk,
                    title=candidate_title or f"Pasted source {index}",
                    content=chunk,
                    source_type="pasted_text",
                    status="extracted" if len(chunk.split()) >= 25 else "failed",
                    error=(
                        None
                        if len(chunk.split()) >= 25
                        else "Pasted source is too short to analyze. Add the source text."
                    ),
                    extraction_method="pasted text",
                    word_count=len(chunk.split()),
                )
            )

    return sources
