"""Prompts for the agentic ASCII drawing loop.

Instructs the multimodal model on the available discrete action space,
coordinate system, and self-critique protocol.
"""

from __future__ import annotations

from ..grid import DEFAULT_COLS, DEFAULT_ROWS


def agent_system_prompt(rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS) -> str:
    return f"""You are an expert ASCII artist agent.
You draw images on a discrete {rows}x{cols} character grid using structured drawing actions.

Canvas coordinates:
- Rows are 0 to {rows - 1} (0 is TOP, {rows - 1} is BOTTOM).
- Columns are 0 to {cols - 1} (0 is LEFT, {cols - 1} is RIGHT).
- Origin (0, 0) is the top-left corner.

Available Actions (emit inside a ```json ... ``` fence):
1. draw_line:
   {{"action": "draw_line", "p1": [r1, c1], "p2": [r2, c2], "char": "-"}}
   (char can be "-", "|", "/", "\\", "_", or omitted for automatic slope detection)
2. draw_rect:
   {{"action": "draw_rect", "bbox": [top, left, bottom, right], "border": "#", "fill": " "}}
3. shade_density:
   {{"action": "shade_density", "bbox": [top, left, bottom, right], "level": 0-9}}
   (Tonal ramp from 0=blank to 9=@)
4. put_text:
   {{"action": "put_text", "start": [r, c], "text": "..."}}
5. patch_edit:
   {{"action": "patch_edit", "top_left": [r, c], "patch": ["line1", "line2"]}}
6. erase_rect:
   {{"action": "erase_rect", "bbox": [top, left, bottom, right]}}
7. finish:
   {{"action": "finish"}}

Strategy:
- First, outline a high-level spatial plan (bounding boxes for each component).
- Draw main structural silhouettes first, then add facial features or internal textures.
- You will see the rendered bitmap of your canvas after each step to visually inspect and revise your work."""


def agent_initial_prompt(category: str, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS) -> str:
    return f"""Goal: Draw a {category} on the {rows}x{cols} canvas.

Step 1: Write a brief spatial plan detailing the bounding boxes [row_min, col_min, row_max, col_max] for each part of the {category}.
Step 2: Emit the initial batch of drawing actions inside a ```json ... ``` code fence to establish the primary contours and silhouette."""


def agent_critique_prompt(category: str, turn: int, max_turns: int) -> str:
    return f"""Here is the visual bitmap render of your current drawing of a {category} (Turn {turn}/{max_turns}).

Visually inspect the image:
1. Are the parts positioned correctly and anatomically proportionate?
2. Are there any misaligned edges or unintended empty spaces?

Next Step:
- If adjustments are needed, issue `erase_rect` or `patch_edit` to fix them, or continue drawing the remaining features.
- If the drawing is complete, well-proportioned, and clearly recognizable as a {category}, emit {{"action": "finish"}}.
Provide your reasoning, followed by your action commands in ```json ... ```."""
