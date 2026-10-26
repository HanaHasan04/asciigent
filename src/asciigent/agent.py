"""Agentic controller for sequential ASCII art generation with active visual perception.

Implements the multi-turn SketchAgent-style loop:
  1. High-level spatial plan (bounding boxes).
  2. Sequential tool execution on AsciiCanvas.
  3. Visual rendering of intermediate canvas state.
  4. Active perception & critique by multimodal VLM.
  5. Localized refinement until completion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .actions import ActionResult, AsciiCanvas, parse_action_blocks
from .grid import DEFAULT_COLS, DEFAULT_ROWS, AsciiGrid
from .llm import LLMClient
from .prompts.agent import (
    agent_critique_prompt,
    agent_critique_text_prompt,
    agent_initial_prompt,
    agent_no_feedback_prompt,
    agent_system_prompt,
)
from .render import RenderStyle, render_to_png_bytes

FEEDBACK_MODES = ("visual", "text_only", "no_feedback")


@dataclass
class AgentTurn:
    turn: int
    raw_response: str
    actions_executed: list[dict[str, Any]]
    results: list[ActionResult]
    finished: bool
    grid_snapshot: AsciiGrid


@dataclass
class AgentSessionResult:
    category: str
    turns: list[AgentTurn] = field(default_factory=list)
    final_grid: AsciiGrid | None = None
    success: bool = True
    stop_reason: str = "completed"

    def total_actions(self) -> int:
        return sum(len(t.actions_executed) for t in self.turns)


def run_agent_drawing_session(
    client: LLMClient,
    category: str,
    max_turns: int = 4,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    render_style: RenderStyle | None = None,
    sample_index: int = 0,
    feedback: str = "visual",
) -> AgentSessionResult:
    """Run an autonomous agentic drawing session with visual, text-only, or no feedback."""
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    if feedback not in FEEDBACK_MODES:
        raise ValueError(f"unknown feedback mode {feedback!r}; expected one of {FEEDBACK_MODES}")

    canvas = AsciiCanvas(rows=rows, cols=cols)
    system = agent_system_prompt(rows, cols)
    style = render_style or RenderStyle()

    session = AgentSessionResult(category=category)
    history_transcript = ""

    for turn_idx in range(1, max_turns + 1):
        if turn_idx == 1:
            prompt = agent_initial_prompt(category, rows, cols)
            images = []
        else:
            grid_current = canvas.to_grid()
            if feedback == "visual":
                png_bytes = render_to_png_bytes(grid_current, style=style)
                critique = agent_critique_prompt(category, turn=turn_idx, max_turns=max_turns)
                images = [png_bytes]
            elif feedback == "text_only":
                critique = agent_critique_text_prompt(
                    category, grid_current.to_text(), turn=turn_idx, max_turns=max_turns
                )
                images = []
            else:  # no_feedback
                critique = agent_no_feedback_prompt(category, turn=turn_idx, max_turns=max_turns)
                images = []

            prompt = f"Conversation History so far:\n{history_transcript}\n\n{critique}"

        # Call the client with multimodal support
        resp = client.complete(
            prompt=prompt,
            images=images,
            system=system,
            sample_index=sample_index,
        )

        content_text = resp.text
        if resp.empty:
            raise ValueError("agent returned an empty response")
        commands = parse_action_blocks(content_text)
        if not commands:
            raise ValueError("agent returned no parseable drawing actions")

        # Execute actions on the mutable canvas
        results: list[ActionResult] = []
        is_finished = False

        for cmd in commands:
            if cmd.get("action") == "finish":
                is_finished = True
                break
            res = canvas.execute_action(cmd)
            results.append(res)

        grid_current = canvas.to_grid()
        session.turns.append(
            AgentTurn(
                turn=turn_idx,
                raw_response=content_text,
                actions_executed=commands,
                results=results,
                finished=is_finished,
                grid_snapshot=grid_current,
            )
        )

        history_transcript += f"\n--- Turn {turn_idx} ---\n{content_text}\n"
        history_transcript += f"Action execution results: {results!r}\n"

        if is_finished or turn_idx == max_turns:
            session.stop_reason = "finish_action" if is_finished else "max_turns_reached"
            break

    session.final_grid = canvas.to_grid()
    return session
