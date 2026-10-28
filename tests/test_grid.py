"""Round-trip and parse-report tests for the canonical 24x48 AsciiGrid."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asciigent.grid import AsciiGrid, extract_art_block  # noqa: E402

CAT = r"""
 /\_/\
( o.o )
 > ^ <
""".strip("\n")


def test_roundtrip_is_exact():
    """Parsing then re-emitting must not change the art."""
    grid, rep = AsciiGrid.from_text(CAT, rows=6, cols=12, extract=False)
    assert grid.to_text(trim=True).rstrip("\n") == CAT
    assert rep.rows_truncated == 0 and rep.cols_truncated == 0


def test_fence_is_stripped():
    text = "Here you go:\n```\n" + CAT + "\n```\nHope that helps!"
    grid, rep = AsciiGrid.from_text(text, rows=6, cols=12)
    assert rep.used_fence
    assert grid.to_text(trim=True).rstrip("\n") == CAT
    assert "no_code_fence" not in rep.issues


def test_missing_fence_is_flagged():
    _, rep = AsciiGrid.from_text(CAT, rows=6, cols=12)
    assert "no_code_fence" in rep.issues
    assert not rep.clean


def test_padding_reaches_exact_size():
    grid, rep = AsciiGrid.from_text(CAT, rows=10, cols=20, extract=False)
    assert grid.rows == 10 and grid.cols == 20
    assert all(len(row) == 20 for row in grid.cells)
    assert rep.rows_padded == 7
    assert rep.clean, "padding a small drawing is not damage"


def test_oversize_is_truncated_and_reported():
    big = "\n".join(["#" * 60] * 30)
    grid, rep = AsciiGrid.from_text(big, rows=24, cols=48, extract=False)
    assert grid.rows == 24 and grid.cols == 48
    assert rep.rows_truncated == 6 and rep.cols_truncated == 12
    assert "too_many_rows" in rep.issues and "too_many_cols" in rep.issues


def test_tabs_and_unicode_are_repaired_loudly():
    grid, rep = AsciiGrid.from_text("a\tb\n██", rows=4, cols=10, extract=False)
    assert rep.tabs_expanded == 1 and rep.nonascii_replaced == 2
    assert "tabs" in rep.issues and "non_ascii" in rep.issues
    assert grid.cells[1].startswith("??")


def test_crlf_does_not_create_phantom_rows():
    grid, rep = AsciiGrid.from_text("ab\r\ncd", rows=4, cols=4, extract=False)
    assert rep.source_rows == 2
    assert grid.cells[0] == "ab  "


def test_longest_fence_wins():
    text = "```\nx\n```\nand\n```\n" + CAT + "\n```"
    body, used = extract_art_block(text)
    assert used and "o.o" in body


def test_diff_counts_only_changed_cells():
    a, _ = AsciiGrid.from_text(CAT, rows=6, cols=12, extract=False)
    b, _ = AsciiGrid.from_text(CAT.replace("o.o", "O.O"), rows=6, cols=12, extract=False)
    assert a.diff_count(b) == 2


def test_scramble_cells_preserves_histogram_only():
    grid, _ = AsciiGrid.from_text(CAT, rows=6, cols=12, extract=False)
    s = grid.scrambled("cells", seed=1)
    assert s.char_histogram() == grid.char_histogram()
    assert s != grid


def test_scramble_ink_preserves_silhouette():
    grid, _ = AsciiGrid.from_text(CAT, rows=6, cols=12, extract=False)
    s = grid.scrambled("ink", seed=1)
    blanks = lambda g: [[ch == " " for ch in row] for row in g.cells]  # noqa: E731
    assert blanks(s) == blanks(grid)
    assert s.ink_ratio() == grid.ink_ratio()
