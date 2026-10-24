import pytest
from asciigent.grid import AsciiGrid
from asciigent.metrics_vision import _resolve_style
from asciigent.render import RenderStyle


def test_resolve_style():
    s1 = _resolve_style("white_on_black")
    assert s1.fg == (255, 255, 255)
    assert s1.bg == (0, 0, 0)

    s2 = _resolve_style("black_on_white")
    assert s2.fg == (0, 0, 0)
    assert s2.bg == (255, 255, 255)

    custom = RenderStyle(font_size=20)
    assert _resolve_style(custom) is custom
