"""Offline unit tests for the cached, zero-budget-enforced inference client."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")

from asciigent.llm import (  # noqa: E402
    PRICES,
    LLMClient,
    LLMResponse,
    PaidProviderRefused,
    Usage,
    _cache_key,
)
from asciigent.providers import PRESETS  # noqa: E402


def key_for(**over):
    base = {
        "provider": "gemini", "model": "gemini-3.8-flash", "system": None,
        "prompt": "draw a cat", "images": [], "max_tokens": 8000,
        "effort": None, "temperature": None, "sample_index": 0,
    }
    return _cache_key(base | over)


# -- the budget guard -------------------------------------------------------


def test_metered_provider_is_refused_by_default(tmp_path):
    with pytest.raises(PaidProviderRefused, match="zero-budget"):
        LLMClient(provider="anthropic", cache_dir=tmp_path)


def test_metered_provider_needs_an_explicit_opt_in(tmp_path):
    client = LLMClient(provider="anthropic", cache_dir=tmp_path,
                       enabled=False, allow_paid=True)
    assert client.provider_name == "anthropic"


def test_the_default_provider_is_free():
    from asciigent.llm import DEFAULT_PROVIDER

    assert not PRESETS[DEFAULT_PROVIDER].metered


def test_free_providers_report_zero_cost(tmp_path):
    client = LLMClient(cache_dir=tmp_path, enabled=False)
    usage = Usage(input_tokens=10_000_000, output_tokens=10_000_000)
    assert usage.cost_usd(client.model) == 0.0
    assert client.summary()["free_tier"] is True


# -- costing (still exercised, for the metered escape hatch) ----------------


def test_cost_matches_price_table():
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert usage.cost_usd("claude-opus-5") == pytest.approx(30.0)


def test_cache_reads_are_a_tenth_of_input():
    assert Usage(cache_read_tokens=1_000_000).cost_usd("claude-opus-5") == pytest.approx(0.5)


def test_every_priced_model_is_priced_both_ways():
    assert all(len(v) == 2 and all(p > 0 for p in v) for v in PRICES.values())


def test_usage_adds():
    total = Usage(1, 2) + Usage(10, 20, 30, 40)
    assert (total.input_tokens, total.output_tokens) == (11, 22)
    assert (total.cache_read_tokens, total.cache_write_tokens) == (30, 40)


# -- caching ----------------------------------------------------------------


def test_cache_key_is_order_independent_but_content_sensitive():
    a = _cache_key({"model": "m", "prompt": "p", "max_tokens": 1})
    b = _cache_key({"max_tokens": 1, "prompt": "p", "model": "m"})
    c = _cache_key({"model": "m", "prompt": "p!", "max_tokens": 1})
    assert a == b and a != c


def test_provider_and_images_change_the_cache_key():
    assert key_for() != key_for(provider="openrouter")
    assert key_for() != key_for(images=["abc"])


def test_disabled_client_never_calls_out(tmp_path):
    client = LLMClient(cache_dir=tmp_path, enabled=False)
    with pytest.raises(RuntimeError, match="cache miss"):
        client.complete("draw a cat")


def test_cache_hit_is_free_and_flagged(tmp_path):
    client = LLMClient(cache_dir=tmp_path, enabled=False)
    client._store(
        key_for(),
        LLMResponse(text="```\n/\\_/\\\n```", model="gemini-3.8-flash",
                    provider="gemini", usage=Usage(10, 20)),
    )
    got = client.complete("draw a cat")
    assert got.cached and got.cost_usd == 0.0
    assert "/\\_/\\" in got.text
    assert client.n_cache_hits == 1


def test_repeated_samples_do_not_collapse_into_one_cache_entry(tmp_path):
    """n-sample cells must make n real calls, or variance is silently zero."""
    client = LLMClient(cache_dir=tmp_path, enabled=False)
    client._store(key_for(), LLMResponse(text="first draw", model="gemini-3.8-flash"))

    assert client.complete("draw a cat", sample_index=0).text == "first draw"
    with pytest.raises(RuntimeError, match="cache miss"):
        client.complete("draw a cat", sample_index=1)


# -- sampling ---------------------------------------------------------------


def test_temperature_refused_on_models_that_reject_it(tmp_path):
    client = LLMClient(cache_dir=tmp_path, enabled=False)
    with pytest.raises(ValueError, match="rejects temperature"):
        client.complete("draw a cat", model="claude-opus-5", temperature=0.9)


def test_temperature_allowed_on_free_models(tmp_path):
    """Most OpenAI-compatible endpoints accept it, so it must reach the cache."""
    client = LLMClient(cache_dir=tmp_path, enabled=False)
    with pytest.raises(RuntimeError, match="cache miss"):
        client.complete("draw a cat", temperature=0.9)


# -- thinking-model guards --------------------------------------------------


def test_tiny_max_tokens_is_raised_to_the_provider_floor(tmp_path):
    """A 16-token cap on a thinking model returns an empty string, not a
    short answer, because reasoning is billed against the same budget."""
    from asciigent.providers import PRESETS

    client = LLMClient(cache_dir=tmp_path, enabled=False)
    floor = PRESETS["gemini"].min_output_tokens
    assert floor >= 1024

    # The bump happens before the cache key is built, so a tiny request and a
    # floor-sized one are the same cell.
    client._store(key_for(max_tokens=floor), LLMResponse(text="hi", model="m"))
    assert client.complete("draw a cat", max_tokens=16).text == "hi"


def test_truncation_is_flagged_not_written_into_the_text():
    """A marker appended to the body corrupts every downstream parse."""
    resp = LLMResponse(text="cat", model="m", truncated=True)
    assert resp.text == "cat"
    assert resp.truncated and not resp.empty


def test_empty_response_is_detectable():
    assert LLMResponse(text="   ", model="m").empty
    assert not LLMResponse(text="cat", model="m").empty
