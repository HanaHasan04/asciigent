"""Grid -> bitmap.

P2 hinges on the model critiquing a *rendered* drawing rather than the raw
string, so the render is part of the experiment, not a debugging convenience.
Characters are drawn cell-by-cell at exact grid coordinates: we never rely on
the font's own advance width to keep columns aligned, which keeps the output
identical across machines and fonts.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from .grid import AsciiGrid

#: Searched in order; first hit wins.  Consolas/DejaVu/Menlo cover win/linux/mac.
FONT_CANDIDATES = (
    r"C:\Windows\Fonts\consola.ttf",
    r"C:\Windows\Fonts\cour.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Menlo.ttc",
)


@dataclass(frozen=True)
class RenderStyle:
    """Everything that affects the pixels.  Log it with the run."""

    font_size: int = 16
    padding: int = 8
    fg: tuple[int, int, int] = (0, 0, 0)
    bg: tuple[int, int, int] = (255, 255, 255)
    font_path: str | None = None
    #: Extra vertical breathing room between rows, in pixels.
    line_gap: int = 0

    def to_dict(self) -> dict:
        return {
            "font_size": self.font_size,
            "padding": self.padding,
            "fg": list(self.fg),
            "bg": list(self.bg),
            "font_path": self.font_path or find_font(),
            "line_gap": self.line_gap,
        }


def find_font(path: str | None = None) -> str | None:
    """First available monospace font, or None to fall back to PIL's bitmap font."""
    if path:
        return path
    env = os.environ.get("ASCIIGENT_FONT")
    if env and os.path.exists(env):
        return env
    for cand in FONT_CANDIDATES:
        if os.path.exists(cand):
            return cand
    return None


def _load_font(style: RenderStyle):
    path = find_font(style.font_path)
    if path is None:
        return ImageFont.load_default(), True
    return ImageFont.truetype(path, style.font_size), False


def cell_size(font, fallback: bool, style: RenderStyle) -> tuple[int, int]:
    """Pixel size of one character cell."""
    if fallback:
        box = font.getbbox("M")
        return box[2] - box[0], (box[3] - box[1]) + style.line_gap
    # Advance width of a monospace glyph is the same for every character.
    width = round(font.getlength("M")) or style.font_size
    ascent, descent = font.getmetrics()
    return width, ascent + descent + style.line_gap


def render(grid: AsciiGrid, style: RenderStyle | None = None) -> Image.Image:
    """Draw a grid onto a new RGB image."""
    style = style or RenderStyle()
    font, fallback = _load_font(style)
    cw, ch = cell_size(font, fallback, style)

    img = Image.new(
        "RGB",
        (grid.cols * cw + 2 * style.padding, grid.rows * ch + 2 * style.padding),
        style.bg,
    )
    draw = ImageDraw.Draw(img)
    for r, row in enumerate(grid.cells):
        y = style.padding + r * ch
        for c, char in enumerate(row):
            if char == " ":
                continue
            draw.text((style.padding + c * cw, y), char, font=font, fill=style.fg)
    return img


def render_to_png_bytes(grid: AsciiGrid, style: RenderStyle | None = None) -> bytes:
    buf = io.BytesIO()
    render(grid, style).save(buf, format="PNG")
    return buf.getvalue()


def render_to_base64(grid: AsciiGrid, style: RenderStyle | None = None) -> str:
    """PNG as base64 -- the form vision APIs want."""
    return base64.b64encode(render_to_png_bytes(grid, style)).decode("ascii")


def save_png(grid: AsciiGrid, path: str, style: RenderStyle | None = None) -> str:
    render(grid, style).save(path, format="PNG")
    return path
