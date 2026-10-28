"""Judge prompts.

The judge is shown **only the rendered image**, never the character string.
That is the entire point: Wang et al. found VLMs judging ASCII art weight
character-level text over global visual pattern, so a judge that can read the
characters is not measuring whether the picture looks like anything.

Two questions, deliberately kept apart:

- *identify* asks what the picture shows without naming the target. It cannot
  be led, so it is the stronger measure.
- *rate* names the target and asks for a graded score. Useful resolution, but
  it tells the judge the answer, so it inflates.

Never ask both in one call -- a rating that follows the judge's own naming is
contaminated by it.
"""

from __future__ import annotations

IDENTIFY_SYSTEM = """You are shown pictures drawn with text characters on a grid.
Judge the picture as a picture: what the shapes and lines depict, not what any
letters spell."""

IDENTIFY_USER = """What single thing does this picture depict?

Answer with a short noun phrase of one to three words, lowercase, and nothing
else. If it does not depict any recognizable thing, answer exactly: unclear"""

RATE_SYSTEM = IDENTIFY_SYSTEM

RATE_USER = """How recognizable is this picture as a {category}?

Use this scale:
5 - unmistakably a {category}
4 - clearly a {category}, with flaws
3 - plausibly a {category}, but ambiguous
2 - barely suggests a {category}
1 - not a {category} at all, or nothing recognizable

Answer with the single digit and nothing else."""


def identify_messages() -> tuple[str, str]:
    return IDENTIFY_SYSTEM, IDENTIFY_USER


def rate_messages(category: str) -> tuple[str, str]:
    return RATE_SYSTEM, RATE_USER.format(category=category)
