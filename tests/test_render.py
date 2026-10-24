"""The render is what the agent sees in P2, so its geometry must be exact."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asciigent.grid import AsciiGrid  # noqa: E402
from asciigent.render import RenderStyle, cell_size, render, render_to_png_bytes  # noqa: E402
from asciigent.render import _load_font  # noqa: E402


def test_image_size_follows_grid_exactly():
    style = RenderStyle(font_size=16, padding=8)
    grid = AsciiGrid.blank(24, 48)
    font, fallback = _load_font(style)
    cw, ch = cell_size(font, fallback, style)
    img = render(grid, style)
    assert img.size == (48 * cw + 16, 24 * ch + 16)


def test_blank_grid_renders_blank():
    img = render(AsciiGrid.blank(4, 8))
    assert img.getcolors() == [(img.size[0] * img.size[1], (255, 255, 255))]


def test_ink_produces_dark_pixels():
    grid, _ = AsciiGrid.from_text("####\n####", rows=2, cols=4, extract=False)
    img = render(grid).convert("L")
    dark = sum(1 for p in img.get_flattened_data() if p < 128)
    assert dark > 0


def test_render_is_deterministic():
    grid, _ = AsciiGrid.from_text("/\\_/\\", rows=3, cols=8, extract=False)
    assert render_to_png_bytes(grid) == render_to_png_bytes(grid)


def test_columns_align_regardless_of_character():
    """Two rows with ink in the same column must light up the same x pixels."""
    grid, _ = AsciiGrid.from_text("  #  \n  i  ", rows=2, cols=5, extract=False)
    img = render(grid, RenderStyle(padding=0)).convert("L")
    w, h = img.size
    px = img.load()
    cols_with_ink = {x for x in range(w) for y in range(h) if px[x, y] < 128}
    # All ink sits inside the third cell of five.
    assert min(cols_with_ink) >= 2 * (w // 5)
    assert max(cols_with_ink) < 3 * (w // 5)
