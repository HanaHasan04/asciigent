"""P1 prompting arms: zero-shot, few-shot-with-exemplars, procedural.

The three arms differ in exactly one thing -- what help the model gets before
drawing -- so the canvas contract and the output format are shared verbatim.
Anything that differs between arms other than the help itself is a confound.

SYSTEM is deliberately frozen: it is the cacheable prefix, so keep volatile
content (the category) out of it.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..grid import DEFAULT_COLS, DEFAULT_ROWS, AsciiGrid

DATA = Path(__file__).resolve().parents[3] / "data"

ARMS = ("zero_shot", "few_shot", "procedural")


def system_prompt(rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS) -> str:
    return f"""You draw pictures using only ASCII characters.

Canvas rules, which never change:
- The canvas is exactly {rows} rows tall and {cols} columns wide.
- Use only printable ASCII (space through ~). No Unicode, no box-drawing
  characters, no emoji, no tabs.
- Use spaces for blank areas. Leading spaces are meaningful -- they position
  the drawing, so never strip them.
- Output the drawing inside a single fenced code block and nothing else after
  it. No caption, no explanation inside the fence.
- Draw the shape of the thing. Do not spell its name."""


def _fence(body: str) -> str:
    return f"```\n{body}\n```"


def load_exemplars(names: list[str] | None = None) -> dict[str, str]:
    """Read exemplar drawings from ``data/exemplars/<name>.txt``.

    Only ``exemplar_only`` categories may be used here: they are disjoint from
    the held-out set, so a few-shot prompt can never leak its own test target.
    """
    meta = json.loads((DATA / "categories.json").read_text(encoding="utf-8"))
    allowed = set(meta["exemplar_only"])
    names = names or meta["exemplar_only"]

    illegal = set(names) - allowed
    if illegal:
        raise ValueError(
            f"{sorted(illegal)} are not exemplar_only categories -- using them "
            "as few-shot exemplars would leak the test set."
        )

    out = {}
    for name in names:
        path = DATA / "exemplars" / f"{name}.txt"
        if path.exists():
            out[name] = path.read_text(encoding="ascii").rstrip("\n")
    return out


def zero_shot(category: str, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS) -> str:
    return f"Draw a {category} on the {rows}x{cols} canvas."


def few_shot(
    category: str,
    exemplars: dict[str, str] | None = None,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
) -> str:
    """Same request, preceded by worked examples of the format and style."""
    exemplars = load_exemplars() if exemplars is None else exemplars
    if category in exemplars:
        raise ValueError(f"{category!r} cannot be both the target and an exemplar")

    blocks = [
        f"Here is a {name} drawn on this canvas:\n\n{_fence(art)}"
        for name, art in sorted(exemplars.items())
    ]
    return (
        "\n\n".join(blocks)
        + f"\n\nNow draw a {category} on the same {rows}x{cols} canvas, "
        "in the same style."
    )


def procedural(category: str, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS) -> str:
    """Force an explicit spatial plan before any characters are emitted.

    If drawing ability is really spatial reasoning rather than recall, making
    the model lay out parts and extents first should help most on the held-out
    categories -- which is the P1 hypothesis this arm exists to test.
    """
    return f"""Draw a {category} on the {rows}x{cols} canvas.

Work in two steps.

Step 1 -- plan, in plain prose before the code block:
  a. List the parts of a {category} you will draw.
  b. For each part, give the rows and columns it will occupy, as
     `part: rows A-B, cols C-D`. Keep every extent inside the canvas.
  c. State which characters carry which kind of edge: verticals, horizontals,
     the two diagonals, and any fill.

Step 2 -- draw it, following your own plan."""


def build(arm: str, category: str, **kw) -> str:
    """Dispatch by arm name so a sweep can loop over :data:`ARMS`."""
    if arm == "zero_shot":
        return zero_shot(category, **kw)
    if arm == "few_shot":
        return few_shot(category, **kw)
    if arm == "procedural":
        return procedural(category, **kw)
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def exemplar_from_grid(grid: AsciiGrid) -> str:
    """Trim a stored grid for use inside a prompt (trailing blanks are noise)."""
    return grid.to_text(trim=True).rstrip("\n")
