"""Judge logic, tested offline.

The scoring rules are where a judge quietly becomes wrong -- a loose match
inflates accuracy, a missed control gap hides an invalid judge -- so they are
tested without any model in the loop.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asciigent.judge import (  # noqa: E402
    JudgeResult,
    Verdict,
    matches,
    normalize,
    parse_rating,
)


def test_normalize_strips_articles_and_punctuation():
    assert normalize("  A Cat.  ") == "cat"
    assert normalize("the hot air balloon!") == "hot air balloon"


def test_exact_and_substring_matches():
    assert matches("cat", "cat")
    assert matches("a cat sitting down", "cat")
    assert matches("Teapot.", "teapot")


def test_declared_synonyms_count():
    assert matches("kitten", "cat")
    assert matches("bike", "bicycle")
    assert matches("balloon", "hot air balloon")


def test_wrong_answers_and_unclear_do_not_count():
    assert not matches("dog", "cat")
    assert not matches("unclear", "cat")
    assert not matches("", "cat")
    assert not matches("some scribbles", "octopus")


def test_ratings_are_parsed_from_noisy_replies():
    assert parse_rating("4") == 4
    assert parse_rating("  Rating: 2\n") == 2
    assert parse_rating("I would say 5 out of 5") == 5
    assert parse_rating("no digits here") is None
    assert parse_rating("") is None


def test_control_gap_detects_a_judge_reading_characters():
    """A judge that likes scrambled images as much as real ones is invalid."""
    good = JudgeResult("cat", {
        "real": Verdict("real", "cat", True, 5),
        "scramble_cells": Verdict("scramble_cells", "unclear", False, 1),
    })
    bad = JudgeResult("cat", {
        "real": Verdict("real", "cat", True, 4),
        "scramble_cells": Verdict("scramble_cells", "cat", True, 4),
    })
    assert good.control_gap() == 4
    assert bad.control_gap() == 0


def test_control_gap_is_none_without_controls():
    assert JudgeResult("cat", {"real": Verdict("real", "cat", True, 5)}).control_gap() is None


def test_result_dict_exposes_the_columns_the_analysis_needs():
    r = JudgeResult("cat", {
        "real": Verdict("real", "a cat", True, 4),
        "scramble_cells": Verdict("scramble_cells", "unclear", False, 1),
    })
    d = r.to_dict()
    assert d["judge_correct"] is True
    assert d["judge_score"] == 4
    assert d["judge_control_gap"] == 3
