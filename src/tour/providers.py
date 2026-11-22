"""Replaceable LLM providers. A Chat-Completions-compatible HTTP adapter and a
scripted fixture for tests. Keys are read from environment variables on the
server only; nothing here logs Authorization headers or key material.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Protocol

import httpx


@dataclass
class ProviderResult:
    content: str
    provider: str
    model: str
    usage: dict | None = None
    latency_seconds: float = 0.0
    finish_reason: str | None = None


class ProviderError(Exception):
    """Typed provider failure; ``kind`` drives the user-facing message."""

    def __init__(self, kind: str, message: str, status_code: int | None = None):
        super().__init__(message)
        self.kind = kind  # timeout | network | http_401 | http_429 | http_5xx | parse
        self.status_code = status_code


class LLMProvider(Protocol):
    async def generate(
        self, messages: list[dict], output_schema: dict, limits: dict
    ) -> ProviderResult: ...


def strip_code_fence(text: str) -> str:
    """Remove at most one complete markdown code fence around the JSON."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            inner = "\n".join(lines[1:-1])
            return inner.strip()
    return text


def parse_strict_json(content: str) -> dict:
    """Strict JSON parse; no eval, no heuristic repair of broken JSON."""
    import json

    try:
        value = json.loads(strip_code_fence(content), parse_constant=_reject_constant)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ProviderError("parse", f"model returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ProviderError("parse", "model output is not a JSON object")
    return value


def _reject_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


class ChatCompletionsProvider:
    """Minimal adapter for Chat-Completions-compatible HTTP APIs.

    Supports ``text_json`` (plain text with a JSON body) and ``json_object``
    (``response_format`` sent only when configured). Other protocols should
    become separate providers, not extra fields mixed into this one.
    """

    def __init__(self, *, base_url, model, api_key, response_mode="text_json",
                 request_timeout_seconds=45.0, max_output_tokens=1500,
                 temperature=0.2, client=None):
        self.base_url = str(base_url).rstrip("/")
        self.model = model
        self.api_key = api_key
        self.response_mode = response_mode
        self.request_timeout_seconds = request_timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._client = client
        self._owns_client = client is None

    def _client_for(self):
        if self._client is not None:
            return self._client
        return httpx.AsyncClient(timeout=httpx.Timeout(self.request_timeout_seconds))

    async def generate(self, messages: list[dict], output_schema: dict, limits: dict) -> ProviderResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": limits.get("max_output_tokens", self.max_output_tokens),
            "temperature": limits.get("temperature", self.temperature),
        }
        if self.response_mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        started = time.monotonic()
        client = self._client_for()
        try:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
            except httpx.TimeoutException as exc:
                raise ProviderError("timeout", f"provider timed out: {exc}") from exc
            except httpx.HTTPError as exc:
                raise ProviderError("network", f"provider request failed: {exc}") from exc
            latency = time.monotonic() - started
            if response.status_code == 401:
                raise ProviderError("http_401", "provider rejected the API key (401)", 401)
            if response.status_code == 429:
                raise ProviderError("http_429", "provider rate limited (429)", 429)
            if response.status_code >= 500:
                raise ProviderError("http_5xx", f"provider error ({response.status_code})", response.status_code)
            if response.status_code != 200:
                raise ProviderError(
                    "http", f"provider returned HTTP {response.status_code}", response.status_code
                )
            try:
                body = response.json()
            except ValueError as exc:
                raise ProviderError("parse", "provider response is not JSON") from exc
            choices = body.get("choices") or []
            if not choices:
                raise ProviderError("parse", "provider response has no choices")
            content = (choices[0].get("message") or {}).get("content")
            if not isinstance(content, str) or not content.strip():
                raise ProviderError("parse", "provider returned empty content")
            return ProviderResult(
                content=content,
                provider="chat_completions_compatible",
                model=body.get("model") or self.model,
                usage=body.get("usage") if isinstance(body.get("usage"), dict) else None,
                latency_seconds=latency,
                finish_reason=choices[0].get("finish_reason"),
            )
        finally:
            if self._owns_client:
                await client.aclose()


class FakeProvider:
    """Scripted fixture for tests and local demos (never a silent fallback)."""

    def __init__(self, script: list, provider="fixture", model="fixture-model"):
        self.script = list(script)
        self.calls = 0
        self.provider = provider
        self.model = model

    async def generate(self, messages: list[dict], output_schema: dict, limits: dict) -> ProviderResult:
        if not self.script:
            raise ProviderError("http_429", "fixture script exhausted")
        entry = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(entry, ProviderError):
            raise entry
        if isinstance(entry, str):
            return ProviderResult(
                content=entry, provider=self.provider, model=self.model,
                usage={"fixture": True}, latency_seconds=0.01,
            )
        return ProviderResult(
            content=entry["content"], provider=self.provider, model=self.model,
            usage=entry.get("usage"), latency_seconds=0.01,
            finish_reason=entry.get("finish_reason"),
        )
