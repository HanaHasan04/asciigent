"""Token accounting and response shape, shared by client and providers.

Separate module so ``providers`` and ``llm`` can both use these without a
circular import.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

#: USD per 1M tokens, (input, output), for metered providers only.
#: Free-tier and locally-hosted models are absent and cost exactly nothing.
PRICES = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def cost_usd(self, model: str) -> float:
        inp, out = PRICES.get(model, (0.0, 0.0))
        return (
            self.input_tokens * inp
            + self.cache_read_tokens * inp * 0.1
            + self.cache_write_tokens * inp * 1.25
            + self.output_tokens * out
        ) / 1_000_000

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str = "unknown"
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None
    cached: bool = False
    latency_s: float = 0.0
    request_id: str | None = None
    #: Hit the output cap. On thinking models this often means the reasoning
    #: consumed the whole budget and ``text`` is empty -- a failed sample, not
    #: a short one. Kept as a flag so ``text`` stays clean for parsing.
    truncated: bool = False

    @property
    def empty(self) -> bool:
        return not self.text.strip()

    @property
    def cost_usd(self) -> float:
        return 0.0 if self.cached else self.usage.cost_usd(self.model)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cost_usd"] = self.cost_usd
        return d
