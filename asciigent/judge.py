"""Model-as-judge over rendered drawings with structure-destroying scrambled controls.

To mitigate the text-reading bias of vision-language models on ASCII art
(Wang et al., 2025), the judge inspects only rendered PNG bitmaps and compares
every real drawing against two scrambled controls:

- **scramble_cells**: uniform random permutation of all characters across the
  full ``24 x 48`` grid.
- **scramble_ink**: random permutation of non-blank characters within the
  occupied silhouette mask.

The control gap ``real_score - max(scramble_cells, scramble_ink)`` isolates
genuine geometric recognizability from character-distribution shortcuts.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .grid import AsciiGrid
from .llm import LLMClient
from .prompts.judge import identify_messages, rate_messages
from .render import RenderStyle, render_to_png_bytes

#: Accept a near-miss name as correct. Kept small and explicit -- a generous
#: synonym list silently inflates accuracy, so anything added here is a
#: methodological choice that belongs in the write-up.
SYNONYMS: dict[str, set[str]] = {
    "cat": {"kitten", "kitty", "feline"},
    "dog": {"puppy", "hound", "canine"},
    "bear": {"teddy bear", "grizzly"},
    "horse": {"pony", "stallion", "mare"},
    "octopus": {"squid"},
    "giraffe": set(),
    "snail": {"slug"},
    "penguin": set(),
    "lighthouse": {"light house"},
    "cactus": {"succulent"},
    "teapot": {"tea pot", "kettle"},
    "bicycle": {"bike"},
    "mushroom": {"toadstool"},
    "hot air balloon": {"balloon", "air balloon"},
}

UNCLEAR = "unclear"

#: Deliberately far larger than the few words we want back. Thinking models
#: spend this budget on reasoning first and emit the answer last, so a tight
#: cap yields an empty response. Depth is controlled by ``effort``, not here.
JUDGE_MAX_TOKENS = 2048


@dataclass
class Verdict:
    """One judgement of one image."""

    variant: str           # real | scramble_cells | scramble_ink
    identified: str = ""   # the judge's free-text answer
    correct: bool = False  # did it match the true category
    rating: int | None = None
    raw_rating: str = ""
    #: The model returned nothing usable. Distinct from a low score: a failed
    #: call is missing data and must be excluded, not counted as a bad drawing.
    failed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class JudgeResult:
    category: str
    verdicts: dict[str, Verdict] = field(default_factory=dict)

    @property
    def real(self) -> Verdict | None:
        return self.verdicts.get("real")

    def control_gap(self) -> float | None:
        """Real rating minus the best control rating.

        Positive and large means the judge is responding to structure. Near
        zero or negative means the judge is not measuring shape, and every
        score in the sweep should be treated as void until that is fixed.
        """
        real = self.verdicts.get("real")
        controls = [
            v.rating for k, v in self.verdicts.items()
            if k != "real" and v.rating is not None
        ]
        if real is None or real.rating is None or not controls:
            return None
        return real.rating - max(controls)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "judge_identified": self.real.identified if self.real else "",
            "judge_correct": bool(self.real and self.real.correct),
            "judge_score": self.real.rating if self.real else None,
            "judge_control_gap": self.control_gap(),
            "verdicts": {k: v.to_dict() for k, v in self.verdicts.items()},
        }


def normalize(answer: str) -> str:
    text = answer.strip().lower()
    text = re.sub(r"^(a|an|the)\s+", "", text)
    return re.sub(r"[^a-z ]", "", text).strip()


def matches(answer: str, category: str) -> bool:
    """Does a free-text identification name the target category?"""
    got = normalize(answer)
    if not got or got == UNCLEAR:
        return False
    want = category.lower()
    if got == want or want in got:
        return True
    return any(s and (got == s or s in got) for s in SYNONYMS.get(want, set()))


def parse_rating(text: str) -> int | None:
    """Pull a 1-5 score out of the judge's reply."""
    m = re.search(r"[1-5]", text)
    return int(m.group()) if m else None


def judge_image(
    client: LLMClient,
    png: bytes,
    category: str,
    variant: str,
    style: RenderStyle | None = None,
    sample_index: int = 0,
) -> Verdict:
    """Two separate calls: identify without the answer, then rate with it.

    ``max_tokens`` is generous even though both answers are a few words long.
    On a thinking model the reasoning is charged against the output cap, so a
    tight cap returns an empty string -- the answer never gets emitted. Depth
    is limited with ``effort`` instead, which is what the cap was for.
    """
    sys_i, user_i = identify_messages()
    ident = client.complete(
        user_i, images=[png], system=sys_i, max_tokens=JUDGE_MAX_TOKENS,
        effort="low", sample_index=sample_index,
    )
    answer = ident.text.strip().splitlines()[0] if ident.text.strip() else ""

    sys_r, user_r = rate_messages(category)
    rated = client.complete(
        user_r, images=[png], system=sys_r, max_tokens=JUDGE_MAX_TOKENS,
        effort="low", sample_index=sample_index,
    )
    return Verdict(
        variant=variant,
        identified=answer,
        correct=matches(answer, category),
        rating=parse_rating(rated.text),
        raw_rating=rated.text.strip()[:40],
        failed=ident.empty or rated.empty,
    )


def judge_drawing(
    client: LLMClient,
    grid: AsciiGrid,
    category: str,
    style: RenderStyle | None = None,
    controls: tuple[str, ...] = ("cells",),
    seed: int = 0,
) -> JudgeResult:
    """Judge the drawing and its scrambled controls.

    ``controls`` defaults to one scramble, because each control doubles the
    call count and free-tier requests are the real budget. Run both
    (``("cells", "ink")``) on a calibration subset rather than the whole sweep.
    """
    result = JudgeResult(category=category)
    result.verdicts["real"] = judge_image(
        client, render_to_png_bytes(grid, style), category, "real"
    )
    for mode in controls:
        scrambled = grid.scrambled(mode, seed=seed)
        result.verdicts[f"scramble_{mode}"] = judge_image(
            client, render_to_png_bytes(scrambled, style), category,
            f"scramble_{mode}",
        )
    return result
