"""Prompts for the agentic ASCII drawing loop (P2: draft -> render -> perceive -> revise).

Instructs the multimodal model on the available discrete action space,
coordinate system, and visual self-critique protocol.
"""

from __future__ import annotations

from ..grid import DEFAULT_COLS, DEFAULT_ROWS


def agent_system_prompt(rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS) -> str:
    return f"""You are an expert ASCII artist agent.
You draw and iteratively refine expressive ASCII art on a {rows}x{cols} character grid through a multi-turn draft -> render -> perceive -> revise loop.

Canvas coordinates:
- Rows are 0 to {rows - 1} (0 is TOP, {rows - 1} is BOTTOM).
- Columns are 0 to {cols - 1} (0 is LEFT, {cols - 1} is RIGHT).
- Center of the canvas is approximately row {rows // 2}, col {cols // 2}.

Available Actions (emit as a JSON array inside a ```json ... ``` fence):
1. patch_edit (PRIMARY TOOL for drawing and revising multi-line ASCII shapes):
   {{"action": "patch_edit", "top_left": [r, c], "patch": ["line1", "line2", ...]}}
2. put_text (place a single-line feature at [r, c]):
   {{"action": "put_text", "start": [r, c], "text": "..."}}
3. draw_line:
   {{"action": "draw_line", "p1": [r1, c1], "p2": [r2, c2], "char": "/"}}
4. erase_rect:
   {{"action": "erase_rect", "bbox": [top, left, bottom, right]}}
5. draw_rect:
   {{"action": "draw_rect", "bbox": [top, left, bottom, right], "border": "-", "fill": " "}}
6. shade_density:
   {{"action": "shade_density", "bbox": [top, left, bottom, right], "level": 0-9}}
7. finish:
   {{"action": "finish"}}

Artistic Rules:
- Use expressive ASCII contour characters: / \\ _ | ( ) - ^ o = v ~ ' .
- NEVER draw animals as boxy '#' rectangles, and NEVER spell the category word as text.
- On Turn 1, draw a clean initial structural draft (silhouette of head and body) centered around col 12..32 using `patch_edit`.
- On subsequent turns, critique your rendered drawing and issue targeted `patch_edit` actions to fix proportions and add missing details (whiskers, paws, tail, ears, shading) at exact [row, col] coordinates."""


def agent_initial_prompt(category: str, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS) -> str:
    return f"""Goal: Draw a recognizable {category} on the {rows}x{cols} canvas (Turn 1: Initial Draft).

Step 1: Briefly state the row and column placement for the {category}'s main silhouette (head and body outline, centered around rows 4..16, cols 14..30).
Step 2: Emit a `patch_edit` action inside a ```json ... ``` code fence to place the initial draft outline on the canvas. Leave finer details (such as extended whiskers, curled tail, or ground/collar details) for the visual refinement turn after you inspect the rendered bitmap."""


def agent_critique_prompt(
    category: str,
    turn: int,
    max_turns: int,
    numbered_grid: str = "",
) -> str:
    grid_section = (
        f"\nCoordinate reference of current canvas:\n```\n{numbered_grid}\n```\n"
        if numbered_grid
        else ""
    )
    return f"""Here is the rendered bitmap image of your current drawing of a {category} (Turn {turn}/{max_turns}).
{grid_section}
Visually inspect the rendered bitmap image:
1. Perception & Critique: What visual features of the {category} are missing, incomplete, or asymmetrical in the rendered image (e.g., whiskers, paws/feet, tail, inner ear detail, eye/nose alignment)?
2. Revision Plan: Specify the exact [row, col] coordinates from the coordinate ruler where you will add or refine these parts.

Next Step:
- If this is Turn {turn} and details can be improved or added, issue `patch_edit` or `put_text` actions at the exact coordinates to refine the drawing (do NOT emit `finish` in the same batch as edits so you can inspect the result).
- Only if the drawing is already complete, detailed, and well-balanced, emit `[{{"action": "finish"}}]`.
Provide your visual critique followed by your JSON action block in ```json ... ```."""


def agent_critique_text_prompt(category: str, ascii_text: str, turn: int, max_turns: int) -> str:
    return f"""Here is the raw character text grid of your current drawing of a {category} (Turn {turn}/{max_turns}):

```
{ascii_text}
```

Inspect the character grid:
1. Are the parts positioned correctly and anatomically proportionate?
2. Are there any misaligned edges or missing features?

Next Step:
- If adjustments are needed, issue `erase_rect` or `patch_edit` to fix them, or continue drawing the remaining features.
- If the drawing is complete, well-proportioned, and clearly recognizable as a {category}, emit {{"action": "finish"}}.
Provide your reasoning, followed by your action commands in ```json ... ```."""


def agent_no_feedback_prompt(category: str, turn: int, max_turns: int) -> str:
    return f"""Continue drawing the {category} (Turn {turn}/{max_turns}).

Emit your next set of drawing actions to add features or complete the drawing.
If you have completed all parts of the {category}, emit {{"action": "finish"}}.
Provide your action commands in ```json ... ```."""
