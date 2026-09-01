from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"


class OpenAIAPIError(RuntimeError):
    """Raised when the OpenAI Responses API call fails."""


def load_env_file(path: str | Path = ".env") -> dict[str, str]:
    """Load simple KEY=VALUE entries from a .env file.

    The project intentionally avoids depending on python-dotenv. This parser is
    conservative and does not perform shell interpolation.
    """

    target = Path(path)
    if not target.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        values[key] = value
    return values


def resolve_secret(name: str, env_file: str | Path = ".env") -> str | None:
    """Resolve a secret from the process environment or a local .env file."""

    value = os.environ.get(name)
    if value:
        return value
    return load_env_file(env_file).get(name) or None


def extract_output_text(payload: dict[str, Any]) -> str:
    """Extract assistant text from a Responses API JSON payload."""

    direct_text = payload.get("output_text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    parts: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str) and text:
                parts.append(text)

    text = "\n".join(part.strip() for part in parts if part.strip()).strip()
    if text:
        return text

    raise OpenAIAPIError("OpenAI response did not contain output text.")


def extract_usage(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return {}

    result: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            result[key] = value
    return result


def _compact_openai_error(exc: urllib.error.HTTPError) -> str:
    body = exc.read().decode("utf-8", errors="replace")
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body[:800]

    error = data.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        code = error.get("code")
        error_type = error.get("type")
        pieces = [str(piece) for piece in (error_type, code, message) if piece]
        if pieces:
            return " | ".join(pieces)[:800]
    return body[:800]


class OpenAIResponsesClient:
    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = RESPONSES_ENDPOINT,
        timeout: float = 120.0,
        max_retries: int = 2,
        retry_wait_seconds: float = 2.0,
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_wait_seconds = retry_wait_seconds

    def create_response(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        max_output_tokens: int,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": input_text,
            "max_output_tokens": max_output_tokens,
            "store": False,
        }
        if temperature is not None:
            body["temperature"] = temperature

        payload = json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(self._max_retries + 1):
            request = urllib.request.Request(
                self._endpoint,
                data=payload,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or exc.code >= 500
                if retryable and attempt < self._max_retries:
                    time.sleep(self._retry_wait_seconds * (attempt + 1))
                    continue
                detail = _compact_openai_error(exc)
                raise OpenAIAPIError(f"OpenAI API returned HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                if attempt < self._max_retries:
                    time.sleep(self._retry_wait_seconds * (attempt + 1))
                    continue
                raise OpenAIAPIError(f"OpenAI API request failed: {exc.reason}") from exc

        raise OpenAIAPIError("OpenAI API request failed after retries.")
