"""Parallel, source-linked extraction from complete source content."""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import requests

from src.llm.ollama import OllamaError, chat_json, get_model
from src.models.schemas import EvidenceExcerpt, Source, SourceFinding, SourceSummary
from src.research.chunker import chunk_text

CHUNK_CHARACTERS = max(1000, int(os.getenv("RESEARCH_CHUNK_CHARACTERS", "3000")))
SUMMARY_TIMEOUT_SECONDS = max(
    60, int(os.getenv("RESEARCH_SUMMARY_TIMEOUT_SECONDS", "360"))
)
# A 7B local model can become substantially slower when source and chunk pools are
# nested. Keep the safe default serial; users with suitable hardware can opt in.
MAX_CHUNK_WORKERS = max(1, int(os.getenv("RESEARCH_CHUNK_WORKERS", "1")))
MAX_SOURCE_WORKERS = max(1, int(os.getenv("RESEARCH_SOURCE_WORKERS", "1")))
MAX_TRANSIENT_RETRIES = 2
RETRY_DELAYS_SECONDS = (2, 5)
_SUMMARY_CACHE: Dict[str, SourceSummary] = {}
_CHUNK_CACHE: Dict[str, Dict[str, List[str]]] = {}


class SourceSummarizationError(RuntimeError):
    """Raised when source-level extraction fails."""


def _format_provider_error(exc: Exception) -> str:
    if type(exc).__name__ == "NoSessionContext":
        return (
            "Streamlit session context was accessed from a background worker. "
            "Restart the app to load the thread-safe research pipeline."
        )
    status_code = getattr(exc, "status_code", None)
    message = getattr(exc, "message", None) or str(exc).strip()
    if not message:
        message = repr(exc)
    if status_code is not None:
        return f"HTTP {status_code}: {message}"
    return message


def _as_list(payload: Any, key: str) -> List[str]:
    value = payload.get(key, []) if isinstance(payload, dict) else []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()] if isinstance(value, list) else []


def _parse_json_object(content: str) -> Dict[str, Any]:
    """Accept JSON wrapped in markdown/prose without accepting partial JSON."""
    candidate = content.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].lstrip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        payload = None
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            payload = parsed
            break
        if payload is None:
            raise
    if not isinstance(payload, dict):
        raise json.JSONDecodeError("Expected a JSON object", candidate, 0)
    return payload


def _summarize_chunk(
    source: Source,
    source_id: str,
    chunk: str,
    chunk_number: int,
) -> Dict[str, Any]:
    prompt = f"""Extract evidence from chunk {chunk_number} of source {source_id}.
Use only the text provided. Do not infer facts that are not stated. Preserve numbers, dates,
names, limitations, and caveats. Return JSON with exactly these keys:
topic, central_argument, methodology, sample_size, population, date,
verified_facts, important_context, caveats, claims_to_verify, implications.
Also return evidence, an array of objects with claim, excerpt, and category. Every excerpt
must be copied verbatim from the provided text and category must be one of:
verified_fact, source_reported_claim, context, caveat, claim_to_verify.
Keep each array to at most 5 concise items. If a category has no supported items, return [].

SOURCE TITLE: {source.title}
SOURCE URL: {source.url or 'pasted source'}
TEXT:
{chunk}
"""
    cache_key = hashlib.sha256(
        f"{get_model()}:{source_id}:{chunk_number}:{chunk}".encode()
    ).hexdigest()
    if cache_key in _CHUNK_CACHE:
        return _CHUNK_CACHE[cache_key]

    payload: Optional[Dict[str, Any]] = None
    last_json_error: Optional[json.JSONDecodeError] = None
    for attempt in range(MAX_TRANSIENT_RETRIES + 1):
        try:
            if attempt:
                request_prompt = f"""Extract only source-supported evidence from this text.
Return one compact JSON object with exactly these keys:
verified_facts (array of short claims), caveats (array), evidence (array).
Each evidence item must contain claim, excerpt, and category. Copy excerpts exactly from
the text. Return no Markdown or explanation. Keep at most 3 items per array.

TEXT:
{chunk}
"""
            else:
                request_prompt = prompt
            content = chat_json(
                [
                    {"role": "system", "content": "You extract cautious, source-grounded evidence."},
                    {"role": "user", "content": request_prompt},
                ],
                timeout=SUMMARY_TIMEOUT_SECONDS,
                max_tokens=900 if attempt else 700,
                temperature=0,
            )
            try:
                payload = _parse_json_object(content)
                break
            except json.JSONDecodeError as exc:
                last_json_error = exc
                if attempt >= MAX_TRANSIENT_RETRIES:
                    raise SourceSummarizationError(
                        f"Ollama returned incomplete or invalid JSON while analyzing {source_id}."
                    ) from exc
                time.sleep(RETRY_DELAYS_SECONDS[attempt])
        except Exception as exc:
            if isinstance(exc, SourceSummarizationError):
                raise
            status_code = getattr(getattr(exc, "__cause__", None), "response", None)
            status_code = getattr(status_code, "status_code", None)
            transient = status_code in {429, 500, 502, 503, 504} or isinstance(
                exc, (OllamaError, requests.RequestException, TimeoutError)
            )
            if not transient or attempt >= MAX_TRANSIENT_RETRIES:
                raise SourceSummarizationError(
                    f"Could not analyze {source_id} with Ollama: "
                    f"{_format_provider_error(exc)}"
                ) from exc
            delay = RETRY_DELAYS_SECONDS[attempt]
            time.sleep(delay)
    if payload is None:
        raise SourceSummarizationError(
            f"Ollama returned unreadable JSON while analyzing {source_id}."
        ) from last_json_error
    result: Dict[str, Any] = {
        key: _as_list(payload, key)
        for key in ("verified_facts", "important_context", "caveats", "claims_to_verify")
    }
    for key in (
        "topic",
        "central_argument",
        "methodology",
        "sample_size",
        "population",
        "date",
    ):
        value = payload.get(key, "")
        result[key] = value.strip() if isinstance(value, str) else ""
    result["implications"] = _as_list(payload, "implications")
    raw_evidence = payload.get("evidence", [])
    result["evidence"] = [
        item for item in (raw_evidence if isinstance(raw_evidence, list) else [])
        if isinstance(item, dict)
        and isinstance(item.get("claim"), str)
        and isinstance(item.get("excerpt"), str)
        and item["claim"].strip()
        and item["excerpt"].strip()
        and item["excerpt"].strip() in chunk
    ]
    # Qwen sometimes puts a source-supported finding in `evidence` but leaves
    # `verified_facts` empty. Preserve that finding in the source summary rather
    # than dropping it before comparative synthesis.
    for item in result["evidence"]:
        claim = item["claim"].strip()
        category = str(item.get("category", "verified_fact")).strip()
        if category == "verified_fact":
            if claim not in result["verified_facts"]:
                result["verified_facts"].append(claim)
        elif category == "source_reported_claim":
            # Keep source-reported findings in evidence without promoting them
            # to independently verified facts.
            continue
        elif category in {"context", "caveat"}:
            if claim not in result["important_context"]:
                result["important_context"].append(claim)
        elif category == "claim_to_verify" and claim not in result["claims_to_verify"]:
            result["claims_to_verify"].append(claim)
    _CHUNK_CACHE[cache_key] = result
    return result


def summarize_source(
    source: Source,
    source_id: str,
) -> SourceSummary:
    cache_key = hashlib.sha256(
        f"{get_model()}:{source.title}:{source.url}:{source.content}".encode()
    ).hexdigest()
    if cache_key in _SUMMARY_CACHE:
        return _SUMMARY_CACHE[cache_key]

    chunks = chunk_text(source.content, CHUNK_CHARACTERS)
    if not chunks:
        raise SourceSummarizationError(f"{source_id} has no readable content.")

    combined: Dict[str, Any] = {
        "verified_facts": [],
        "important_context": [],
        "caveats": [],
        "claims_to_verify": [],
        "evidence": [],
        "topic": "",
        "central_argument": "",
        "methodology": "",
        "sample_size": "",
        "population": "",
        "date": "",
        "implications": [],
    }
    with ThreadPoolExecutor(max_workers=min(MAX_CHUNK_WORKERS, len(chunks))) as executor:
        futures = [
            executor.submit(
                _summarize_chunk,
                source,
                source_id,
                chunk,
                index,
            )
            for index, chunk in enumerate(chunks, start=1)
        ]
        for future in futures:
            result = future.result()
            for key, values in result.items():
                if isinstance(values, list):
                    for value in values:
                        if value not in combined[key]:
                            combined[key].append(value)
                elif isinstance(values, str) and values and not combined[key]:
                    combined[key] = values

    summary = SourceSummary(
        source_id=source_id,
        title=source.title,
        url=source.url,
        verified_facts=combined["verified_facts"],
        important_context=combined["important_context"],
        caveats=combined["caveats"],
        claims_to_verify=combined["claims_to_verify"],
        evidence=[
            EvidenceExcerpt(
                source_id=source_id,
                title=source.title,
                url=source.url,
                claim=item["claim"].strip(),
                excerpt=item["excerpt"].strip(),
                category=item.get("category", "verified_fact"),
            )
            for item in combined["evidence"]
        ],
        topic=combined["topic"] or source.title,
        central_argument=combined["central_argument"],
        methodology=combined["methodology"],
        sample_size=combined["sample_size"],
        population=combined["population"],
        date=combined["date"],
        implications=combined["implications"],
    )
    summary.findings = [
        SourceFinding(
            source_id=source_id,
            title=source.title,
            url=source.url,
            claim=evidence.claim,
            evidence=[evidence],
            methodology=summary.methodology,
            sample_size=summary.sample_size,
            population=summary.population,
            date=summary.date,
            implication=summary.implications[0] if summary.implications else "",
        )
        for evidence in summary.evidence
        if evidence.category in {"verified_fact", "source_reported_claim"}
    ]
    _SUMMARY_CACHE[cache_key] = summary
    return summary


def summarize_sources(
    sources: List[Source],
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[SourceSummary]:
    results: List[Optional[SourceSummary]] = [None] * len(sources)
    with ThreadPoolExecutor(max_workers=min(MAX_SOURCE_WORKERS, len(sources))) as executor:
        futures = {
            executor.submit(
                summarize_source,
                source,
                f"source-{index}",
            ): index
            for index, source in enumerate(sources, start=1)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index - 1] = future.result()
            except SourceSummarizationError:
                raise
            except Exception as exc:
                raise SourceSummarizationError(
                    f"Could not analyze source-{index}: {_format_provider_error(exc)}"
                ) from exc
            completed += 1
            if progress_callback:
                progress_callback(completed, len(sources), f"Source {index} analyzed")
    return [result for result in results if result is not None]
