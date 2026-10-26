from unittest.mock import MagicMock
from asciigent.agent import run_agent_drawing_session
from asciigent.llm import LLMClient
from asciigent.llm_types import LLMResponse, Usage
import json
import pytest


def test_agent_session_executes_actions_and_finishes():
    # Mock LLMClient that emits drawing commands on turn 1, and finishes on turn 2
    client = MagicMock(spec=LLMClient)

    turn1_text = """
Plan: Draw a simple box.
```json
[
  {"action": "draw_rect", "bbox": [2, 2, 8, 12], "border": "#"}
]
```
"""
    turn2_text = """
The box looks well proportioned.
```json
[
  {"action": "finish"}
]
```
"""
    client.complete.side_effect = [
        LLMResponse(text=turn1_text, model="test-model", usage=Usage()),
        LLMResponse(text=turn2_text, model="test-model", usage=Usage()),
    ]

    session = run_agent_drawing_session(
        client=client,
        category="box",
        max_turns=3,
        rows=10,
        cols=15,
    )

    assert len(session.turns) == 2
    assert session.stop_reason == "finish_action"
    assert session.final_grid is not None
    # Check that rect was drawn
    assert session.final_grid.cells[2][2:13] == "###########"
    assert session.final_grid.cells[8][2:13] == "###########"
    assert client.complete.call_count == 2


def test_p2_runner_persists_run_and_distinguishes_samples(tmp_path, monkeypatch):
    from asciigent.experiments import p2_agent

    client = MagicMock(spec=LLMClient)
    client.complete.return_value = LLMResponse(
        text='```json\n[{"action":"put_text","start":[2,3],"text":"/\\\\"},'
             '{"action":"finish"}]\n```',
        model="test-model", usage=Usage(),
    )
    monkeypatch.setattr(p2_agent, "credentials_present", lambda _: True)
    monkeypatch.setattr(p2_agent, "LLMClient", lambda **_: client)

    assert p2_agent.main([
        "--category", "cat", "--samples", "2", "--max-turns", "1",
        "--runs-dir", str(tmp_path),
    ]) == 0
    records = [json.loads(line) for line in (tmp_path / "index.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert len({r["run_id"] for r in records}) == 2
    assert all(r["phase"] == "p2" and "ink_ratio" in r for r in records)
    assert [call.kwargs["sample_index"] for call in client.complete.call_args_list] == [0, 1]
    for record in records:
        directory = tmp_path / record["run_id"]
        assert (directory / "drawing.png").exists()
        assert (directory / "turn_1.txt").exists()
        step = json.loads((directory / "steps.jsonl").read_text())
        assert step["raw_response"] and step["actions"]


@pytest.mark.parametrize("text", ["", "I have drawn a cat."])
def test_agent_rejects_unusable_responses(text):
    client = MagicMock(spec=LLMClient)
    client.complete.return_value = LLMResponse(text=text, model="test", usage=Usage())
    with pytest.raises(ValueError):
        run_agent_drawing_session(client, "cat")


@pytest.mark.parametrize("mode", ["visual", "text_only", "no_feedback"])
def test_agent_feedback_modes(mode):
    client = MagicMock(spec=LLMClient)
    turn1_text = '```json\n[{"action":"draw_line","p1":[1,1],"p2":[1,5]}]\n```'
    turn2_text = '```json\n[{"action":"finish"}]\n```'
    client.complete.side_effect = [
        LLMResponse(text=turn1_text, model="test", usage=Usage()),
        LLMResponse(text=turn2_text, model="test", usage=Usage()),
    ]

    session = run_agent_drawing_session(
        client=client,
        category="cat",
        max_turns=2,
        feedback=mode,
    )
    assert len(session.turns) == 2
    assert session.stop_reason == "finish_action"

    call2_args = client.complete.call_args_list[1]
    images_passed = call2_args.kwargs.get("images", [])

    if mode == "visual":
        assert len(images_passed) == 1
    elif mode == "text_only":
        assert len(images_passed) == 0
        assert "raw character text grid" in call2_args.kwargs["prompt"]
    elif mode == "no_feedback":
        assert len(images_passed) == 0
        assert "Continue drawing" in call2_args.kwargs["prompt"]
