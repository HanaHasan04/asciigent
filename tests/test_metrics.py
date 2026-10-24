"""Structural metrics catch the failure modes that would inflate P1 scores."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asciigent.grid import AsciiGrid  # noqa: E402
from asciigent.metrics import (  # noqa: E402
    char_entropy,
    coverage,
    degenerate,
    spells_name,
    structural,
)

CAT = " /\\_/\\\n( o.o )\n > ^ <"


def g(text, rows=8, cols=16):
    grid, _ = AsciiGrid.from_text(text, rows=rows, cols=cols, extract=False)
    return grid


def test_spelling_the_word_is_caught():
    assert spells_name(g("   CAT   "), "cat")
    assert spells_name(g("  a cat  "), "cat")
    assert not spells_name(g(CAT), "cat")


def test_multiword_category_is_caught_on_any_word():
    assert spells_name(g("  BALLOON  "), "hot air balloon")
    assert not spells_name(g(CAT), "hot air balloon")


def test_short_words_do_not_trigger_false_positives():
    """'a' in 'hot air balloon' must not match every drawing containing 'a'."""
    assert not spells_name(g(" /\\a/\\ "), "a")


def test_blank_and_solid_canvases_are_degenerate():
    assert degenerate(AsciiGrid.blank(8, 16))
    assert degenerate(g("\n".join(["#" * 16] * 8)))
    assert not degenerate(g(CAT))


def test_coverage_separates_big_drawings_from_postage_stamps():
    big = g("\n".join(["#" + " " * 14 + "#"] * 8))
    small = g("##")
    assert coverage(big) > 0.9
    assert coverage(small) < 0.1


def test_entropy_is_zero_for_a_single_repeated_character():
    assert char_entropy(g("####")) == 0.0
    assert char_entropy(g(CAT)) > 1.0


def test_structural_bundles_everything():
    m = structural(g(CAT), "cat")
    assert not m.spells_name and not m.degenerate
    assert 0 < m.ink_ratio < 1 and m.distinct_chars > 1
    assert set(m.to_dict()) == {
        "ink_ratio", "coverage", "char_entropy",
        "distinct_chars", "spells_name", "degenerate",
    }
