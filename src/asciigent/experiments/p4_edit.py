"""P4 Experiment: Conversational Editing of ASCII Drawings.

Takes an existing ASCII drawing (from a file or a built-in hardcoded example)
plus a natural-language edit instruction, sends both the rendered image and the
numbered text grid to the multimodal LLM, and asks it to produce ``patch_edit``
or ``erase_rect`` actions that implement the requested change.

After editing, the experiment reports how many cells changed and whether the
edit was localized (changed cells mostly within the region named by the
instruction).

Usage::

    python -m asciigent.experiments.p4_edit --instruction "add whiskers"
    python -m asciigent.experiments.p4_edit --drawing path/to/cat.txt \\
        --instruction "make the eyes look wider"
    python -m asciigent.experiments.p4_edit --dry-run --instruction "add a hat"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from asciigent.actions import AsciiCanvas, parse_action_blocks
from asciigent.grid import DEFAULT_COLS, DEFAULT_ROWS, AsciiGrid
from asciigent.llm import DEFAULT_PROVIDER, LLMClient
from asciigent.providers import PRESETS, credentials_present, default_model
from asciigent.render import render_to_png_bytes, save_png

# ---------------------------------------------------------------------------
# Built-in hardcoded drawing (a simple cat face) used when --drawing is absent
# ---------------------------------------------------------------------------
_DEFAULT_DRAWING = r"""
                                                
                                                
         /\_____/\                              
        /  o   o  \                             
       ( ==  ^  == )                            
        )         (                             
       (           )                            
      ( (  )   (  ) )                           
     (__(__)___(__)__)                          
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
""".lstrip("\n")

# ---------------------------------------------------------------------------
# System prompt for the editing LLM call
# ---------------------------------------------------------------------------
EDIT_SYSTEM = (
    "You are editing an existing ASCII drawing. You will see the current drawing "
    "rendered as an image AND as a numbered text grid. Issue only patch_edit or "
    "erase_rect actions to apply the requested change. Do not redraw from scratch. "
    "Minimize changes outside the affected region.\n\n"
    "Actions must be emitted as a JSON array inside a ```json ... ``` code fence.\n"
    "patch_edit schema:  {\"action\": \"patch_edit\", \"top_left\": [r, c], \"patch\": [\"line0\", ...]}\n"
    "erase_rect schema:  {\"action\": \"erase_rect\", \"bbox\": [r1, c1, r2, c2]}"
)


def _build_user_message(grid: AsciiGrid, instruction: str) -> str:
    """Format the numbered grid + edit instruction as a user message."""
    numbered = grid.numbered_text()
    return (
        "Current drawing (numbered grid):\n"
        f"{numbered}\n\n"
        f"Instruction: {instruction}\n\n"
        "Issue drawing actions in ```json [...] ```."
    )


def _load_drawing(path: str | None) -> AsciiGrid:
    """Load a drawing from *path* or fall back to the built-in example."""
    if path:
        text = Path(path).read_text(encoding="ascii", errors="replace")
        grid, _ = AsciiGrid.from_text(text, rows=DEFAULT_ROWS, cols=DEFAULT_COLS, extract=False)
        return grid
    # Hardcoded fallback
    grid, _ = AsciiGrid.from_text(
        _DEFAULT_DRAWING, rows=DEFAULT_ROWS, cols=DEFAULT_COLS, extract=False
    )
    return grid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drawing", default=None,
                    help="Path to a .txt file containing the ASCII drawing to edit. "
                         "If omitted, a built-in cat face is used.")
    ap.add_argument("--instruction", default="add whiskers",
                    help="Natural-language edit instruction, e.g. 'add a hat'.")
    ap.add_argument("--provider", default=DEFAULT_PROVIDER, choices=sorted(PRESETS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print configuration and exit without calling the API.")
    args = ap.parse_args(argv)

    model = args.model or default_model(args.provider)
    runs_dir = Path(args.runs_dir)

    print("P4 Conversational Editing")
    print(f"  instruction : {args.instruction}")
    print(f"  drawing     : {args.drawing or '(built-in cat face)'}")
    print(f"  provider    : {args.provider} [{model}]")
    print(f"  runs-dir    : {runs_dir}")

    if args.dry_run:
        print("\n[dry-run] Loading drawing …")
        grid = _load_drawing(args.drawing)
        print(f"[dry-run] Canvas: {grid.rows}x{grid.cols}")
        print("[dry-run] User message preview:")
        print(_build_user_message(grid, args.instruction)[:500])
        print("\n[dry-run] Configuration validated successfully.")
        return 0

    if not credentials_present(args.provider):
        cfg = PRESETS[args.provider]
        print(f"No credentials found for {args.provider}. Please set ${cfg.key_env}")
        return 2

    # -- Load original drawing ------------------------------------------------
    before_grid = _load_drawing(args.drawing)
    png_before = render_to_png_bytes(before_grid)

    # -- Call LLM -------------------------------------------------------------
    client = LLMClient(provider=args.provider, model=model)
    user_msg = _build_user_message(before_grid, args.instruction)
    resp = client.complete(
        prompt=user_msg,
        images=[png_before],
        system=EDIT_SYSTEM,
    )

    print("\n--- LLM response ---")
    print(resp.text[:2000])

    # -- Parse + apply actions ------------------------------------------------
    commands = parse_action_blocks(resp.text)
    if not commands:
        print("\nNo parseable actions found in response. Nothing changed.")
    else:
        canvas = AsciiCanvas(rows=before_grid.rows, cols=before_grid.cols, base=before_grid)
        for cmd in commands:
            act = cmd.get("action", "")
            if act not in ("patch_edit", "erase_rect"):
                print(f"  Skipping non-edit action: {act!r}")
                continue
            result = canvas.execute_action(cmd)
            print(f"  {act}: {result.cells_modified} cells modified")
        after_grid = canvas.to_grid()

    # -- Diff & summary -------------------------------------------------------
    after_grid = canvas.to_grid() if commands else before_grid
    diff_count = before_grid.diff_count(after_grid)

    # Heuristic localization: check if changed cells cluster in one quarter
    mask = before_grid.diff_mask(after_grid)
    changed_rows = [r for r, row in enumerate(mask) if any(row)]
    row_span = (max(changed_rows) - min(changed_rows) + 1) if changed_rows else 0
    localized = row_span <= (before_grid.rows // 2) if changed_rows else True

    # -- Save before / after PNGs ---------------------------------------------
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_subdir = runs_dir / "p4_edit"
    run_subdir.mkdir(parents=True, exist_ok=True)

    (run_subdir / "before.txt").write_text(before_grid.to_text(), encoding="ascii")
    (run_subdir / "before.png").write_bytes(png_before)
    (run_subdir / "after.txt").write_text(after_grid.to_text(), encoding="ascii")
    save_png(after_grid, str(run_subdir / "after.png"))

    print("\n--- Edit summary ---")
    print(f"  Cells changed  : {diff_count}")
    print(f"  Edit localized : {localized}  (changed rows span={row_span} of {before_grid.rows})")
    print(f"  Saved to       : {run_subdir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
