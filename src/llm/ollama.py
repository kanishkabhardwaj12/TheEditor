"""Small native Ollama client shared by the research and writing workflows."""

import os
from typing import Any, Dict

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_NUM_THREADS = 4


class OllamaError(RuntimeError):
    """Raised when Ollama cannot return a usable response."""


def _chat_url() -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3].rstrip("/")
    return f"{base_url}/api/chat"


def get_model() -> str:
    return os.getenv("OLLAMA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _get_int_setting(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def chat_json(
    messages: list[Dict[str, str]],
    *,
    timeout: int,
    max_tokens: int,
    temperature: float,
) -> str:
    model = get_model()
    url = _chat_url()
    num_threads = _get_int_setting("OLLAMA_NUM_THREADS", DEFAULT_NUM_THREADS)
    num_context = _get_int_setting("OLLAMA_NUM_CTX", 4096)
    try:
        response = requests.post(
            url,
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_thread": num_threads,
                    "num_ctx": num_context,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload: Any = response.json()
        content = payload.get("message", {}).get("content", "")
    except requests.ConnectionError as exc:
        raise OllamaError(
            f"Could not connect to Ollama at {url}. Start Ollama and try again."
        ) from exc
    except requests.Timeout as exc:
        raise OllamaError(
            f"Ollama timed out while running {model}. The model may still be loading; "
            "try again or increase the configured timeout."
        ) from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        detail = ""
        if exc.response is not None:
            try:
                body = exc.response.json()
                detail = str(body.get("error", "")).strip()
            except (ValueError, TypeError):
                detail = exc.response.text.strip()
        suffix = f": {detail}" if detail else ""
        if status == 404:
            raise OllamaError(
                f"Ollama does not have model {model!r}. Run `ollama pull {model}` "
                "and try again."
            ) from exc
        raise OllamaError(f"Ollama returned HTTP {status}{suffix}.") from exc
    except requests.RequestException as exc:
        raise OllamaError(f"Ollama request failed: {exc}") from exc
    except (ValueError, TypeError) as exc:
        raise OllamaError("Ollama returned an invalid response.") from exc

    if not isinstance(content, str) or not content.strip():
        raise OllamaError(f"Ollama returned an empty response for model {model!r}.")
    return content
