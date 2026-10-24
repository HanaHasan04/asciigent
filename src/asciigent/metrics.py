"""Cheap structural metrics -- everything measurable without a judge.

None of these measure whether a drawing *looks like* the thing; that is P3's
job and needs a judge calibrated against humans.  What they do measure is
whether a sample is well-formed enough to be worth judging at all, and they
catch the specific failure modes that would otherwise quietly inflate scores:

- the model writes the word instead of drawing it,
- the model returns an almost-empty or almost-solid canvas,
- the model ignores the format contract.

Report these alongside recognizability, never instead of it.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

from .grid import BLANK, AsciiGrid

#: Below this the canvas is essentially blank; above it, essentially solid.
EMPTY_INK = 0.01
SOLID_INK = 0.80


@dataclass
class StructuralMetrics:
    ink_ratio: float
    coverage: float
    char_entropy: float
    distinct_chars: int
    spells_name: bool
    degenerate: bool

    def to_dict(self) -> dict:
        return asdict(self)


def coverage(grid: AsciiGrid) -> float:
    """Fraction of the canvas spanned by the drawing's bounding box.

    Distinguishes "used the canvas" from "drew a postage stamp in the corner",
    which matters when comparing arms: a prompt that yields bigger drawings is
    not automatically yielding better ones.
    """
    box = grid.bbox()
    if box is None:
        return 0.0
    top, left, bottom, right = box
    return ((bottom - top + 1) * (right - left + 1)) / (grid.rows * grid.cols)


def char_entropy(grid: AsciiGrid) -> float:
    """Shannon entropy (bits) over non-blank characters.

    Near zero means one character repeated -- a fill pattern rather than a
    drawing that uses different marks for different kinds of edge.
    """
    hist = {c: n for c, n in grid.char_histogram().items() if c != BLANK}
    total = sum(hist.values())
    if not total:
        return 0.0
    return -sum((n / total) * math.log2(n / total) for n in hist.values())


def spells_name(grid: AsciiGrid, category: str) -> bool:
    """True if the category name appears as literal text in the drawing.

    Catches the degenerate 'write CAT instead of drawing a cat' response.
    It will not catch a name rendered as large block letters -- that needs the
    judge, and is worth watching for by eye during the pilot.
    """
    text = grid.to_text().lower()
    flat = re.sub(r"\s+", " ", text)
    for word in category.lower().split():
        if len(word) < 3:
            continue
        if word in flat.replace(" ", "") or word in flat:
            return True
    return False


def degenerate(grid: AsciiGrid) -> bool:
    """Blank or near-solid canvases carry no shape information."""
    return not (EMPTY_INK < grid.ink_ratio() < SOLID_INK)


def structural(grid: AsciiGrid, category: str) -> StructuralMetrics:
    return StructuralMetrics(
        ink_ratio=round(grid.ink_ratio(), 4),
        coverage=round(coverage(grid), 4),
        char_entropy=round(char_entropy(grid), 4),
        distinct_chars=len([c for c in grid.char_histogram() if c != BLANK]),
        spells_name=spells_name(grid, category),
        degenerate=degenerate(grid),
    )
