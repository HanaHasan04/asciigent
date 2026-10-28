"""Prompt-arm tests, mostly about leakage -- the one bug that would void P1."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asciigent.prompts.text2ascii import (  # noqa: E402
    ARMS,
    build,
    few_shot,
    load_exemplars,
    system_prompt,
)


def test_exemplars_load_and_fit_the_canvas():
    ex = load_exemplars()
    assert ex, "no exemplar files found in data/exemplars"
    for name, art in ex.items():
        lines = art.split("\n")
        assert len(lines) <= 24, f"{name} is too tall"
        assert max(len(line) for line in lines) <= 48, f"{name} is too wide"
        assert art.isascii(), f"{name} contains non-ASCII"


def test_held_out_categories_can_never_be_exemplars():
    with pytest.raises(ValueError, match="leak"):
        load_exemplars(["octopus"])


def test_target_cannot_appear_in_its_own_few_shot_prompt():
    with pytest.raises(ValueError, match="both the target and an exemplar"):
        few_shot("tree")


def test_few_shot_contains_exemplars_and_the_target():
    prompt = few_shot("octopus")
    assert "octopus" in prompt
    assert "```" in prompt
    assert any(name in prompt for name in load_exemplars())


def test_arms_differ_only_in_the_help_given():
    prompts = {arm: build(arm, "octopus") for arm in ARMS}
    assert len({p for p in prompts.values()}) == 3
    for arm, p in prompts.items():
        assert "octopus" in p, arm


def test_no_arm_carries_a_task_rule_the_others_lack():
    """A rule given to one arm only is a confound, not a prompting style.

    Every constraint on *what a valid drawing is* belongs in the shared system
    prompt; an arm may only differ in the help it offers for producing one.
    """
    shared = system_prompt().lower()
    rules = ["do not spell", "only printable ascii", "fenced code block"]
    for rule in rules:
        assert rule in shared, f"{rule!r} must live in the shared system prompt"
        for arm in ARMS:
            assert rule not in build(arm, "octopus").lower(), (
                f"{arm} restates {rule!r} -- that is an arm-specific rule"
            )


def test_procedural_demands_a_plan_before_the_drawing():
    p = build("procedural", "giraffe")
    assert p.index("plan") < p.index("draw it")
    assert "rows A-B, cols C-D" in p, "the plan must commit to spatial extents"


def test_system_prompt_states_the_canvas_and_stays_category_free():
    s = system_prompt(24, 48)
    assert "24 rows" in s and "48 columns" in s
    assert "cat" not in s.lower().split(), "system prefix must stay cacheable"


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="unknown arm"):
        build("magic", "cat")
