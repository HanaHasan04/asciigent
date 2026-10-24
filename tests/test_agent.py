from unittest.mock import MagicMock
from asciigent.agent import run_agent_drawing_session
from asciigent.llm import LLMClient
from asciigent.llm_types import LLMResponse, Usage


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
