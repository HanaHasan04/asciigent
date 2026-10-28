"""Canonical ASCII-art representation.

Every drawing in ASCIIgent is a rigid ``rows x cols`` matrix of printable ASCII.
Generation, rendering, judging, diffing, and scrambled-character controls all
operate on :class:`AsciiGrid`.

Parsing is lossy-but-loud: any formatting adjustments (truncation, padding, or
missing code fences) are recorded in a :class:`ParseReport` rather than
silently repaired, so experiments can track format compliance directly.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import asdict, dataclass, field

DEFAULT_ROWS = 24
DEFAULT_COLS = 48

#: The 95 printable ASCII characters (0x20 space .. 0x7E tilde) we draw with.
PRINTABLE = frozenset(chr(c) for c in range(0x20, 0x7F))
BLANK = " "

#: Tabs are ambiguous in art; we expand them but always flag it as damage.
TAB_WIDTH = 4

_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)


@dataclass
class ParseReport:
    """What we had to do to model output to make it a grid."""

    source_rows: int = 0
    source_cols: int = 0
    used_fence: bool = False
    tabs_expanded: int = 0
    nonascii_replaced: int = 0
    rows_truncated: int = 0
    cols_truncated: int = 0
    rows_padded: int = 0
    cols_padded: int = 0
    issues: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """True when the model returned exactly a well-formed canvas."""
        return not self.issues

    def to_dict(self) -> dict:
        d = asdict(self)
        d["clean"] = self.clean
        return d


def extract_art_block(text: str) -> tuple[str, bool]:
    """Pull the drawing out of a model response.

    Prefers the longest fenced block, since our prompts require one.  Falls back
    to the whole response (the caller sees ``used_fence=False`` and can treat
    the sample as suspect).
    """
    blocks = _FENCE_RE.findall(text)
    if blocks:
        return max(blocks, key=len), True
    return text, False


class AsciiGrid:
    """A fixed-size character canvas."""

    __slots__ = ("rows", "cols", "cells")

    def __init__(self, cells: list[str], rows: int, cols: int):
        if len(cells) != rows or any(len(r) != cols for r in cells):
            raise ValueError(f"cells do not form a {rows}x{cols} grid")
        self.rows = rows
        self.cols = cols
        self.cells = cells

    # -- construction ----------------------------------------------------

    @classmethod
    def blank(cls, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS) -> "AsciiGrid":
        return cls([BLANK * cols] * rows, rows, cols)

    @classmethod
    def from_text(
        cls,
        text: str,
        rows: int = DEFAULT_ROWS,
        cols: int = DEFAULT_COLS,
        extract: bool = True,
    ) -> tuple["AsciiGrid", ParseReport]:
        """Parse raw model output into a canvas, reporting every repair made."""
        rep = ParseReport()

        body, rep.used_fence = extract_art_block(text) if extract else (text, False)
        if extract and not rep.used_fence:
            rep.issues.append("no_code_fence")

        body = body.replace("\r\n", "\n").replace("\r", "\n").strip("\n")

        lines = body.split("\n")
        rep.source_rows = len(lines)

        cleaned: list[str] = []
        for line in lines:
            if "\t" in line:
                rep.tabs_expanded += line.count("\t")
                line = line.expandtabs(TAB_WIDTH)
            out = []
            for ch in line:
                if ch in PRINTABLE:
                    out.append(ch)
                else:
                    rep.nonascii_replaced += 1
                    out.append("?")
            cleaned.append("".join(out))

        rep.source_cols = max((len(line) for line in cleaned), default=0)
        if rep.tabs_expanded:
            rep.issues.append("tabs")
        if rep.nonascii_replaced:
            rep.issues.append("non_ascii")

        if len(cleaned) > rows:
            rep.rows_truncated = len(cleaned) - rows
            rep.issues.append("too_many_rows")
            cleaned = cleaned[:rows]
        elif len(cleaned) < rows:
            rep.rows_padded = rows - len(cleaned)
            cleaned += [""] * rep.rows_padded

        fitted = []
        for line in cleaned:
            if len(line) > cols:
                rep.cols_truncated = max(rep.cols_truncated, len(line) - cols)
                line = line[:cols]
            elif len(line) < cols:
                rep.cols_padded = max(rep.cols_padded, cols - len(line))
                line = line.ljust(cols, BLANK)
            fitted.append(line)
        if rep.cols_truncated:
            rep.issues.append("too_many_cols")

        return cls(fitted, rows, cols), rep

    # -- access ----------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AsciiGrid) and self.cells == other.cells

    def __getitem__(self, rc: tuple[int, int]) -> str:
        r, c = rc
        return self.cells[r][c]

    def to_text(self, trim: bool = False) -> str:
        """Render back to a string; ``trim`` drops trailing blanks per line."""
        lines = [line.rstrip() for line in self.cells] if trim else list(self.cells)
        return "\n".join(lines)

    def numbered_text(self) -> str:
        """Grid with 2-digit row/column rulers -- for prompts that edit by coordinate."""
        tens = "     " + "".join(str((c // 10) % 10) for c in range(self.cols))
        ones = "     " + "".join(str(c % 10) for c in range(self.cols))
        return "\n".join([tens, ones] + [f"{r:3d} |{row}" for r, row in enumerate(self.cells)])

    def digest(self) -> str:
        return hashlib.sha1(self.to_text().encode("ascii")).hexdigest()[:12]

    # -- measurements ----------------------------------------------------

    def ink_ratio(self) -> float:
        """Fraction of non-blank cells."""
        ink = sum(1 for row in self.cells for ch in row if ch != BLANK)
        return ink / (self.rows * self.cols)

    def char_histogram(self) -> dict[str, int]:
        hist: dict[str, int] = {}
        for row in self.cells:
            for ch in row:
                hist[ch] = hist.get(ch, 0) + 1
        return hist

    def bbox(self) -> tuple[int, int, int, int] | None:
        """(top, left, bottom, right) inclusive bounds of non-blank cells."""
        rs = [r for r, row in enumerate(self.cells) if row.strip()]
        if not rs:
            return None
        cs = [c for row in self.cells for c, ch in enumerate(row) if ch != BLANK]
        return min(rs), min(cs), max(rs), max(cs)

    def diff_mask(self, other: "AsciiGrid") -> list[list[bool]]:
        """Per-cell changed/unchanged mask -- the P4 edit-localization measure."""
        if (self.rows, self.cols) != (other.rows, other.cols):
            raise ValueError("grids differ in size")
        return [
            [a != b for a, b in zip(ra, rb)]
            for ra, rb in zip(self.cells, other.cells)
        ]

    def diff_count(self, other: "AsciiGrid") -> int:
        return sum(sum(row) for row in self.diff_mask(other))

    # -- controls --------------------------------------------------------

    def scrambled(self, mode: str = "cells", seed: int = 0) -> "AsciiGrid":
        """Structure-destroying control for the P3 judge check.

        ``cells``  shuffle every cell -- preserves the character histogram,
                   destroys all spatial structure.
        ``ink``    shuffle only the non-blank characters among the positions
                   they already occupy -- preserves the silhouette, destroys
                   local texture.

        A judge that scores these near the real drawing is reading characters,
        not pictures (the Wang et al. confound).
        """
        rng = random.Random(seed)
        if mode == "cells":
            flat = [ch for row in self.cells for ch in row]
            rng.shuffle(flat)
            rows = [
                "".join(flat[r * self.cols:(r + 1) * self.cols])
                for r in range(self.rows)
            ]
            return AsciiGrid(rows, self.rows, self.cols)
        if mode == "ink":
            ink = [ch for row in self.cells for ch in row if ch != BLANK]
            rng.shuffle(ink)
            it = iter(ink)
            rows = [
                "".join(BLANK if ch == BLANK else next(it) for ch in row)
                for row in self.cells
            ]
            return AsciiGrid(rows, self.rows, self.cols)
        raise ValueError(f"unknown scramble mode: {mode!r}")
