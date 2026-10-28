"""Cached, cost-tracked, provider-agnostic multimodal inference client.

Provides three core guarantees:

1. **Zero-cost enforcement**: Metered providers raise :class:`PaidProviderRefused`
   unless ``allow_paid=True`` is explicitly passed.
2. **Deterministic disk caching**: Responses are cached on disk keyed by the
   full request payload, ensuring reproducible and resumable sweeps.
3. **Rate-limit resilience**: Works with provider-level pacing and quota-aware
   error handling so interrupted sweeps resume cleanly from the cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from .llm_types import PRICES, LLMResponse, Usage
from .providers import PRESETS, default_model, get_provider

#: Free by default. Changing this is a budget decision, not a tuning knob.
DEFAULT_PROVIDER = "gemini"

#: Current models reject temperature/top_p/top_k with a 400.  Most
#: OpenAI-compatible endpoints still accept it, which matters when an
#: experiment needs sampling diversity.
NO_SAMPLING_MODELS = frozenset(
    {
        "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7",
        "claude-sonnet-5", "claude-fable-5", "claude-fable-5-1",
    }
)


class PaidProviderRefused(RuntimeError):
    """Raised when a billable call would have been made."""


def _cache_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LLMClient:
    """Provider-agnostic client with an on-disk response cache."""

    def __init__(
        self,
        provider: str = DEFAULT_PROVIDER,
        model: str | None = None,
        cache_dir: str | os.PathLike = ".cache/llm",
        max_retries: int = 5,
        timeout: float = 300.0,
        enabled: bool = True,
        allow_paid: bool = False,
    ):
        if provider not in PRESETS:
            raise ValueError(f"unknown provider {provider!r}; have {sorted(PRESETS)}")
        if PRESETS[provider].metered and not allow_paid:
            raise PaidProviderRefused(
                f"provider {provider!r} bills per token and this project has a "
                f"zero-budget constraint. Free options: "
                f"{sorted(p for p, c in PRESETS.items() if not c.metered)}. "
                f"Pass allow_paid=True only if you have decided to spend money."
            )

        self.provider_name = provider
        self.model = model or default_model(provider)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        #: Set False in tests/dry runs to make any uncached call raise.
        self.enabled = enabled
        self.allow_paid = allow_paid
        self._max_retries = max_retries
        self._timeout = timeout
        self._provider = None  # built lazily so offline tests need no SDK/key

        self.total_usage = Usage()
        self.n_calls = 0
        self.n_cache_hits = 0

    @property
    def provider(self):
        if self._provider is None:
            self._provider = get_provider(
                self.provider_name,
                max_retries=self._max_retries,
                timeout=self._timeout,
            )
        return self._provider

    # -- cache -----------------------------------------------------------

    def _path(self, key: str) -> Path:
        sub = self.cache_dir / key[:2]
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{key}.json"

    def _load(self, key: str) -> LLMResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return LLMResponse(
            text=raw["text"],
            model=raw["model"],
            provider=raw.get("provider", "unknown"),
            usage=Usage(**raw["usage"]),
            stop_reason=raw.get("stop_reason"),
            cached=True,
            latency_s=raw.get("latency_s", 0.0),
            request_id=raw.get("request_id"),
            truncated=raw.get("truncated", False),
        )

    def _store(self, key: str, resp: LLMResponse) -> None:
        payload = asdict(resp) | {"cached": False}
        self._path(key).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- calling ---------------------------------------------------------

    def complete(
        self,
        prompt: str,
        images: list[bytes] | None = None,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 8000,
        effort: str | None = None,
        temperature: float | None = None,
        cache_system: bool = False,
        sample_index: int = 0,
    ) -> LLMResponse:
        """One request.  ``images`` are raw PNG bytes, shown before the text.

        ``sample_index`` exists only to distinguish repeated draws of the same
        prompt.  Without it the cache would serve draw #1 for every subsequent
        draw, collapsing an n-sample cell into one response and silently
        reporting zero variance.  It is not sent to the API.
        """
        model = model or self.model
        images = images or []

        if temperature is not None and model in NO_SAMPLING_MODELS:
            raise ValueError(f"{model} rejects temperature (HTTP 400)")

        # Thinking models bill reasoning against the output cap, so a small
        # max_tokens returns an empty string instead of a short answer.
        max_tokens = max(max_tokens, PRESETS[self.provider_name].min_output_tokens)

        key = _cache_key(
            {
                "provider": self.provider_name,
                "model": model,
                "system": system,
                "prompt": prompt,
                "images": [hashlib.sha256(b).hexdigest() for b in images],
                "max_tokens": max_tokens,
                "effort": effort,
                "temperature": temperature,
                "sample_index": sample_index,
            }
        )

        hit = self._load(key)
        if hit is not None:
            self.n_cache_hits += 1
            return hit

        if not self.enabled:
            raise RuntimeError(f"cache miss with client disabled (key {key[:12]})")

        start = time.time()
        raw = self.provider.complete(
            prompt=prompt,
            model=model,
            images=images,
            system=system,
            max_tokens=max_tokens,
            effort=effort,
            temperature=temperature,
            cache_system=cache_system,
        )
        elapsed = time.time() - start

        resp = LLMResponse(
            text=raw.text,
            model=model,
            provider=self.provider_name,
            usage=raw.usage,
            stop_reason=raw.stop_reason,
            cached=False,
            latency_s=elapsed,
            request_id=raw.request_id,
        )
        # Flag truncation rather than editing the text: a marker appended to
        # the body corrupts every downstream parse (grids, judge answers).
        resp.truncated = raw.stop_reason in ("max_tokens", "length")

        self.total_usage = self.total_usage + raw.usage
        self.n_calls += 1
        self._store(key, resp)
        return resp

    # -- reporting -------------------------------------------------------

    def summary(self) -> dict:
        return {
            "provider": self.provider_name,
            "model": self.model,
            "calls": self.n_calls,
            "cache_hits": self.n_cache_hits,
            "usage": asdict(self.total_usage),
            "cost_usd": round(self.total_usage.cost_usd(self.model), 4),
            "free_tier": not PRESETS[self.provider_name].metered,
        }


__all__ = [
    "LLMClient",
    "LLMResponse",
    "PaidProviderRefused",
    "Usage",
    "PRICES",
    "DEFAULT_PROVIDER",
    "_cache_key",
]
