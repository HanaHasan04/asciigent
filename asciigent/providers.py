"""Model providers.

Two backends cover everything we need:

- ``AnthropicProvider`` -- the official Anthropic SDK.
- ``OpenAICompatProvider`` -- the OpenAI wire protocol, which Gemini,
  OpenRouter, Groq and a local vLLM/Ollama server all speak. One class, four
  presets.

Free tiers are rate-limited far more tightly than they are quota-limited, so
every provider paces itself to a configured requests-per-minute. Without that,
a 126-cell sweep spends most of its life in 429 backoff.
"""

from __future__ import annotations

import base64
import os
import re
import threading
import time
from dataclasses import dataclass, field

from .llm_types import Usage


@dataclass
class RawResponse:
    text: str
    usage: Usage
    stop_reason: str | None = None
    request_id: str | None = None


class DailyQuotaExhausted(RuntimeError):
    """The per-day free-tier allowance for this model is used up.

    Distinct from a per-minute limit: waiting will not help today, so a sweep
    must stop immediately rather than grind through its remaining cells
    collecting identical failures.
    """


def _retry_delay(message: str) -> float | None:
    """Pull Google's suggested retry delay out of a 429 message."""
    m = re.search(r"retry in ([0-9.]+)s", message, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"'retryDelay':\s*'(\d+)s'", message)
    return float(m.group(1)) if m else None


def _is_per_day(message: str) -> bool:
    return "PerDay" in message or "per day" in message.lower()


class _Pacer:
    """Blocks so calls stay under a requests-per-minute ceiling.

    Shared across threads so a parallel sweep still respects one global limit.
    """

    def __init__(self, rpm: float | None):
        self.min_interval = 60.0 / rpm if rpm else 0.0
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if not self.min_interval:
            return
        with self._lock:
            delta = time.monotonic() - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.monotonic()


@dataclass
class ProviderConfig:
    name: str
    default_model: str
    base_url: str | None = None
    key_env: str = ""
    #: Free-tier requests per minute. None means unpaced.
    rpm: float | None = None
    #: Providers that bill per token; others report zero cost honestly.
    metered: bool = False
    extra_headers: dict[str, str] = field(default_factory=dict)
    #: Thinking models spend max_tokens on reasoning before emitting anything,
    #: so a small cap returns an empty string rather than a short answer.
    #: Requests below this floor are raised to it.
    min_output_tokens: int = 1
    #: Accepts OpenAI-style ``reasoning_effort`` to control thinking depth.
    supports_reasoning_effort: bool = False


PRESETS: dict[str, ProviderConfig] = {
    "anthropic": ProviderConfig(
        name="anthropic",
        default_model="claude-opus-5",
        key_env="ANTHROPIC_API_KEY",
        metered=True,
    ),
    # Free tier is Flash-only. The 2.5 series retires 2026-10-16 -- do not pin it.
    # Gemini 3 models always think, and thought tokens are charged against
    # max_output_tokens, so short caps yield empty responses.
    "gemini": ProviderConfig(
        name="gemini",
        default_model="gemini-3.8-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_env="GEMINI_API_KEY",
        rpm=10,
        min_output_tokens=2048,
        supports_reasoning_effort=True,
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        default_model="qwen/qwen3-vl-30b-a3b:free",
        base_url="https://openrouter.ai/api/v1",
        key_env="OPENROUTER_API_KEY",
        rpm=18,
        min_output_tokens=1024,
        supports_reasoning_effort=True,
    ),
    "groq": ProviderConfig(
        name="groq",
        default_model="openai/gpt-oss-120b",
        base_url="https://api.groq.com/openai/v1",
        key_env="GROQ_API_KEY",
        rpm=28,
    ),
    # vLLM / Ollama on your own GPU. No key, no pacing.
    "local": ProviderConfig(
        name="local",
        default_model="Qwen/Qwen3-VL-8B-Instruct",
        base_url=os.environ.get("ASCIIGENT_LOCAL_URL", "http://localhost:8000/v1"),
        key_env="ASCIIGENT_LOCAL_KEY",
    ),
}


class Provider:
    """Interface: turn a prompt plus images into text and a token count."""

    def __init__(self, config: ProviderConfig, max_retries: int = 5, timeout: float = 300.0):
        self.config = config
        self.max_retries = max_retries
        self.timeout = timeout
        self.pacer = _Pacer(config.rpm)

    @property
    def name(self) -> str:
        return self.config.name

    def complete(self, **kw) -> RawResponse:  # pragma: no cover - interface
        raise NotImplementedError


class AnthropicProvider(Provider):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        import anthropic

        self._sdk = anthropic
        self._client = anthropic.Anthropic(
            max_retries=self.max_retries, timeout=self.timeout
        )

    def complete(
        self,
        prompt: str,
        model: str,
        images: list[bytes] | None = None,
        system: str | None = None,
        max_tokens: int = 8000,
        effort: str | None = None,
        temperature: float | None = None,
        cache_system: bool = False,
    ) -> RawResponse:
        content: list[dict] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(b).decode("ascii"),
                },
            }
            for b in (images or [])
        ]
        content.append({"type": "text", "text": prompt})

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if system is not None:
            kwargs["system"] = (
                [{"type": "text", "text": system,
                  "cache_control": {"type": "ephemeral"}}]
                if cache_system
                else system
            )
        if effort is not None:
            kwargs["output_config"] = {"effort": effort}
        if temperature is not None:
            kwargs["temperature"] = temperature

        self.pacer.wait()
        msg = self._client.messages.create(**kwargs)
        return RawResponse(
            text="".join(b.text for b in msg.content if b.type == "text"),
            usage=Usage(
                input_tokens=msg.usage.input_tokens,
                output_tokens=msg.usage.output_tokens,
                cache_read_tokens=getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
            ),
            stop_reason=msg.stop_reason,
            request_id=msg._request_id,
        )


class OpenAICompatProvider(Provider):
    """Gemini, OpenRouter, Groq, and any local vLLM/Ollama server."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        from openai import OpenAI

        key = os.environ.get(self.config.key_env) or "not-needed"
        # max_retries=0 on purpose. The SDK retries every 429 blindly, and on a
        # 20-request daily allowance that turns one call into six wasted ones.
        # Retrying is decided below, per failure kind.
        self._client = OpenAI(
            api_key=key,
            base_url=self.config.base_url,
            max_retries=0,
            timeout=self.timeout,
            default_headers=self.config.extra_headers or None,
        )

    def _send(self, kwargs: dict):
        """Call the endpoint, retrying only what is worth retrying.

        - per-day quota  -> give up immediately, the day is over
        - per-minute cap -> wait the delay the server asked for, then retry
        - 503 / network  -> short backoff, these are transient under load
        """
        import openai

        delay = 2.0
        for attempt in range(self.max_retries + 1):
            self.pacer.wait()
            try:
                return self._client.chat.completions.create(**kwargs)
            except openai.RateLimitError as exc:
                text = str(exc)
                if _is_per_day(text):
                    raise DailyQuotaExhausted(text) from None
                wait = _retry_delay(text) or delay
                if attempt == self.max_retries:
                    raise
                time.sleep(min(wait + 1.0, 90.0))
            except (openai.InternalServerError, openai.APIConnectionError,
                    openai.APITimeoutError):
                if attempt == self.max_retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise RuntimeError("unreachable")

    def complete(
        self,
        prompt: str,
        model: str,
        images: list[bytes] | None = None,
        system: str | None = None,
        max_tokens: int = 8000,
        effort: str | None = None,
        temperature: float | None = None,
        cache_system: bool = False,
    ) -> RawResponse:
        content: list[dict] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,"
                    + base64.standard_b64encode(b).decode("ascii")
                },
            }
            for b in (images or [])
        ]
        content.append({"type": "text", "text": prompt})

        messages: list[dict] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if effort is not None and self.config.supports_reasoning_effort:
            # Our scale is wider than OpenAI's; clamp the top end.
            kwargs["reasoning_effort"] = {
                "xhigh": "high", "max": "high"
            }.get(effort, effort)

        resp = self._send(kwargs)
        choice = resp.choices[0]
        usage = resp.usage
        return RawResponse(
            text=choice.message.content or "",
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            stop_reason=choice.finish_reason,
            request_id=getattr(resp, "id", None),
        )


def get_provider(name: str, **kw) -> Provider:
    if name not in PRESETS:
        raise ValueError(f"unknown provider {name!r}; have {sorted(PRESETS)}")
    config = PRESETS[name]
    cls = AnthropicProvider if name == "anthropic" else OpenAICompatProvider
    return cls(config, **kw)


def default_model(provider: str) -> str:
    return PRESETS[provider].default_model


def credentials_present(provider: str) -> bool:
    config = PRESETS[provider]
    return provider == "local" or bool(os.environ.get(config.key_env))
