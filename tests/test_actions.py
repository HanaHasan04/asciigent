import pytest
from asciigent.actions import AsciiCanvas, DENSITY_RAMP, parse_action_blocks
from asciigent.grid import BLANK, AsciiGrid


def test_canvas_blank():
    canvas = AsciiCanvas(rows=10, cols=20)
    grid = canvas.to_grid()
    assert grid.rows == 10
    assert grid.cols == 20
    assert grid.ink_ratio() == 0.0


def test_canvas_draw_line_horizontal_vertical():
    canvas = AsciiCanvas(rows=10, cols=20)
    res_h = canvas.draw_line(2, 5, 2, 10)
    assert res_h.success
    assert res_h.cells_modified == 6
    assert canvas.to_grid().cells[2][5:11] == "------"

    res_v = canvas.draw_line(3, 5, 6, 5)
    assert res_v.success
    assert res_v.cells_modified == 4
    for r in range(3, 7):
        assert canvas.get_cell(r, 5) == "|"


def test_canvas_draw_rect():
    canvas = AsciiCanvas(rows=10, cols=20)
    res = canvas.draw_rect(1, 2, 4, 6, border_char="#", fill_char=".")
    assert res.success
    grid = canvas.to_grid()
    # Check top and bottom border
    assert grid.cells[1][2:7] == "#####"
    assert grid.cells[4][2:7] == "#####"
    # Check interior
    assert grid.cells[2][2:7] == "#...#"
    assert grid.cells[3][2:7] == "#...#"


def test_canvas_shade_density():
    canvas = AsciiCanvas(rows=10, cols=20)
    canvas.shade_density(0, 0, 1, 1, level=9)
    grid = canvas.to_grid()
    assert grid.cells[0][:2] == "@@"
    assert grid.cells[1][:2] == "@@"


def test_canvas_patch_edit_and_erase():
    canvas = AsciiCanvas(rows=10, cols=20)
    patch = ["/\\_/\\", "(o.o)"]
    res = canvas.patch_edit(3, 4, patch)
    assert res.success
    grid = canvas.to_grid()
    assert grid.cells[3][4:9] == "/\\_/\\"
    assert grid.cells[4][4:9] == "(o.o)"

    # Erase a portion
    canvas.erase_rect(3, 4, 3, 8)
    grid_after = canvas.to_grid()
    assert grid_after.cells[3][4:9] == "     "
    assert grid_after.cells[4][4:9] == "(o.o)"


def test_canvas_bounds_clipping():
    canvas = AsciiCanvas(rows=10, cols=10)
    # Drawing outside bounds should not crash
    res = canvas.draw_line(-5, -5, 15, 15, ch="X")
    assert res.success
    # Ensure cells within bounds were updated
    assert canvas.get_cell(0, 0) == "X"
    assert canvas.get_cell(9, 9) == "X"


def test_parse_action_blocks():
    text = """
Here is my plan to draw a house:
```json
[
  {"action": "draw_rect", "bbox": [5, 10, 15, 25], "border": "#"},
  {"action": "put_text", "start": [12, 16], "text": "DOOR"}
]
```
Done!
"""
    cmds = parse_action_blocks(text)
    assert len(cmds) == 2
    assert cmds[0]["action"] == "draw_rect"
    assert cmds[1]["action"] == "put_text"
