"""Discrete action space for agentic ASCII art generation.

Instead of forcing a language model to emit an entire 24x48 matrix in one shot
(which suffers from autoregressive scanline blindness and coordinate drift),
the agent interacts with an :class:`AsciiCanvas` via structured drawing tools:
lines, contours, tonal shading ramps, stamps, and localized patch edits.

All actions operate within the canvas boundaries and clip safely.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .grid import BLANK, DEFAULT_COLS, DEFAULT_ROWS, PRINTABLE, AsciiGrid

# Calibrated tonal luminance ramp from lowest to highest character density
DENSITY_RAMP = " .:-=+*#%@"


@dataclass
class ActionResult:
    action: str
    success: bool
    cells_modified: int = 0
    message: str = ""


class AsciiCanvas:
    """A mutable 2D canvas that executes discrete spatial drawing commands."""

    def __init__(self, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS, base: AsciiGrid | None = None):
        self.rows = rows
        self.cols = cols
        if base is not None:
            if (base.rows, base.cols) != (rows, cols):
                raise ValueError("base grid dimensions mismatch")
            self._grid: list[list[str]] = [list(r) for r in base.cells]
        else:
            self._grid = [[BLANK] * cols for _ in range(rows)]
        self.history: list[dict[str, Any]] = []

    def to_grid(self) -> AsciiGrid:
        """Produce an immutable AsciiGrid snapshot."""
        return AsciiGrid(["".join(r) for r in self._grid], self.rows, self.cols)

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _sanitize_char(self, ch: str) -> str:
        if not ch or ch not in PRINTABLE:
            return "?"
        return ch

    # -- drawing primitives --------------------------------------------------

    def set_cell(self, r: int, c: int, ch: str) -> bool:
        if self.in_bounds(r, c):
            clean = self._sanitize_char(ch)
            if self._grid[r][c] != clean:
                self._grid[r][c] = clean
                return True
        return False

    def get_cell(self, r: int, c: int) -> str:
        if self.in_bounds(r, c):
            return self._grid[r][c]
        return BLANK

    def draw_line(
        self,
        r1: int,
        c1: int,
        r2: int,
        c2: int,
        ch: str | None = None,
    ) -> ActionResult:
        """Draw a Bresenham line between (r1, c1) and (r2, c2).

        If ``ch`` is None, selects an edge character automatically according to slope.
        """
        dr = r2 - r1
        dc = c2 - c1

        if ch is None:
            if dr == 0:
                ch = "-"
            elif dc == 0:
                ch = "|"
            elif (dr > 0 and dc > 0) or (dr < 0 and dc < 0):
                ch = "\\"
            else:
                ch = "/"

        clean_ch = self._sanitize_char(ch)
        modified = 0

        # Bresenham's algorithm
        r, c = r1, c1
        step_r = 1 if dr > 0 else (-1 if dr < 0 else 0)
        step_c = 1 if dc > 0 else (-1 if dc < 0 else 0)
        abs_dr = abs(dr)
        abs_dc = abs(dc)

        if abs_dc >= abs_dr:
            err = 2 * abs_dr - abs_dc
            for _ in range(abs_dc + 1):
                if self.set_cell(r, c, clean_ch):
                    modified += 1
                if err > 0:
                    r += step_r
                    err -= 2 * abs_dc
                err += 2 * abs_dr
                c += step_c
        else:
            err = 2 * abs_dc - abs_dr
            for _ in range(abs_dr + 1):
                if self.set_cell(r, c, clean_ch):
                    modified += 1
                if err > 0:
                    c += step_c
                    err -= 2 * abs_dr
                err += 2 * abs_dc
                r += step_r

        self.history.append({"action": "draw_line", "p1": [r1, c1], "p2": [r2, c2], "char": clean_ch})
        return ActionResult("draw_line", True, modified)

    def draw_rect(
        self,
        r1: int,
        c1: int,
        r2: int,
        c2: int,
        border_char: str | None = None,
        fill_char: str | None = None,
    ) -> ActionResult:
        """Draw an axis-aligned rectangle from (r1, c1) to (r2, c2)."""
        top, bottom = min(r1, r2), max(r1, r2)
        left, right = min(c1, c2), max(c1, c2)
        modified = 0

        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                is_border = (r in (top, bottom)) or (c in (left, right))
                if is_border and border_char is not None:
                    if self.set_cell(r, c, border_char):
                        modified += 1
                elif not is_border and fill_char is not None:
                    if self.set_cell(r, c, fill_char):
                        modified += 1

        self.history.append({
            "action": "draw_rect",
            "bbox": [top, left, bottom, right],
            "border": border_char,
            "fill": fill_char,
        })
        return ActionResult("draw_rect", True, modified)

    def shade_density(self, r1: int, c1: int, r2: int, c2: int, level: int) -> ActionResult:
        """Fill a region with a tonal density level (0 to 9)."""
        idx = max(0, min(level, len(DENSITY_RAMP) - 1))
        fill_ch = DENSITY_RAMP[idx]
        top, bottom = min(r1, r2), max(r1, r2)
        left, right = min(c1, c2), max(c1, c2)
        modified = 0

        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if self.set_cell(r, c, fill_ch):
                    modified += 1

        self.history.append({
            "action": "shade_density",
            "bbox": [top, left, bottom, right],
            "level": idx,
            "char": fill_ch,
        })
        return ActionResult("shade_density", True, modified)

    def put_text(self, r: int, c: int, text: str) -> ActionResult:
        """Write a string horizontally starting at (r, c)."""
        modified = 0
        for i, ch in enumerate(text):
            if self.set_cell(r, c + i, ch):
                modified += 1
        self.history.append({"action": "put_text", "start": [r, c], "text": text})
        return ActionResult("put_text", True, modified)

    def patch_edit(self, top_left_r: int, top_left_c: int, patch_lines: Sequence[str]) -> ActionResult:
        """Replace a subgrid starting at (top_left_r, top_left_c)."""
        modified = 0
        for dr, line in enumerate(patch_lines):
            r = top_left_r + dr
            for dc, ch in enumerate(line):
                c = top_left_c + dc
                if self.set_cell(r, c, ch):
                    modified += 1
        self.history.append({
            "action": "patch_edit",
            "top_left": [top_left_r, top_left_c],
            "patch": list(patch_lines),
        })
        return ActionResult("patch_edit", True, modified)

    def erase_rect(self, r1: int, c1: int, r2: int, c2: int) -> ActionResult:
        """Clear rectangular area back to space."""
        top, bottom = min(r1, r2), max(r1, r2)
        left, right = min(c1, c2), max(c1, c2)
        modified = 0
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if self.set_cell(r, c, BLANK):
                    modified += 1
        self.history.append({"action": "erase_rect", "bbox": [top, left, bottom, right]})
        return ActionResult("erase_rect", True, modified)

    # -- action dispatch -----------------------------------------------------

    def execute_action(self, cmd: dict[str, Any]) -> ActionResult:
        """Execute a parsed JSON action command."""
        act = cmd.get("action")
        if not act:
            return ActionResult("unknown", False, message="missing 'action' key")

        try:
            if act == "draw_line":
                p1 = cmd["p1"]
                p2 = cmd["p2"]
                ch = cmd.get("char")
                return self.draw_line(p1[0], p1[1], p2[0], p2[1], ch=ch)

            if act == "draw_rect":
                b = cmd["bbox"]
                return self.draw_rect(b[0], b[1], b[2], b[3], border_char=cmd.get("border"), fill_char=cmd.get("fill"))

            if act == "shade_density":
                b = cmd["bbox"]
                return self.shade_density(b[0], b[1], b[2], b[3], cmd.get("level", 0))

            if act == "put_text":
                s = cmd["start"]
                return self.put_text(s[0], s[1], cmd["text"])

            if act == "patch_edit":
                tl = cmd["top_left"]
                return self.patch_edit(tl[0], tl[1], cmd["patch"])

            if act == "erase_rect":
                b = cmd["bbox"]
                return self.erase_rect(b[0], b[1], b[2], b[3])

            return ActionResult(act, False, message=f"unknown action {act!r}")
        except Exception as e:
            return ActionResult(act, False, message=str(e))


def parse_action_blocks(response_text: str) -> list[dict[str, Any]]:
    """Extract action commands from model text.

    Supports either a JSON array of commands inside ```json fences or raw JSON.
    """
    json_blocks = re.findall(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", response_text, re.DOTALL)
    commands: list[dict[str, Any]] = []

    for block in json_blocks:
        try:
            data = json.loads(block)
            if isinstance(data, list):
                commands.extend(d for d in data if isinstance(d, dict))
            elif isinstance(data, dict):
                if "actions" in data and isinstance(data["actions"], list):
                    commands.extend(d for d in data["actions"] if isinstance(d, dict))
                elif "action" in data:
                    commands.append(data)
        except json.JSONDecodeError:
            continue

    if not commands:
        # Fallback: scan lines for individual JSON objects
        for line in response_text.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and "action" in data:
                        commands.append(data)
                except json.JSONDecodeError:
                    pass

    return commands
