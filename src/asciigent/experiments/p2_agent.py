"""P2 Experiment: Autonomous Agentic ASCII Generation with Active Perception.

Runs the multi-turn SketchAgent-style loop (plan -> draw -> render -> critique -> refine)
and logs intermediate canvas snapshots for visualization and analysis.

Usage:
    python -m asciigent.experiments.p2_agent --category cat --max-turns 4
    python -m asciigent.experiments.p2_agent --split seen --samples 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from asciigent.agent import run_agent_drawing_session
from asciigent.grid import DEFAULT_COLS, DEFAULT_ROWS
from asciigent.llm import DEFAULT_PROVIDER, LLMClient
from asciigent.metrics import structural
from asciigent.providers import PRESETS, credentials_present, default_model
from asciigent.render import render_to_png_bytes
from asciigent.store import RunStore

DATA = Path(__file__).resolve().parents[3] / "data"


def categories(split: str) -> list[str]:
    meta = json.loads((DATA / "categories.json").read_text(encoding="utf-8"))
    if split == "all":
        return meta["seen"] + meta["held_out"]
    return meta[split]


def main(argv: list[str] | None = None) -> int:
    meta = json.loads((DATA / "categories.json").read_text(encoding="utf-8"))
    rows, cols = meta["canvas"]["rows"], meta["canvas"]["cols"]

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--category", default=None, help="single category override")
    ap.add_argument("--split", default="seen", choices=["seen", "held_out", "all"])
    ap.add_argument("--samples", type=int, default=1, help="draws per category")
    ap.add_argument("--max-turns", type=int, default=4, help="maximum refinement turns")
    ap.add_argument(
        "--feedback", default="visual", choices=["visual", "text_only", "no_feedback"],
        help="control arm: visual (bitmap render), text_only (character grid), or no_feedback",
    )
    ap.add_argument("--provider", default=DEFAULT_PROVIDER, choices=sorted(PRESETS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.samples < 1 or args.max_turns < 1:
        ap.error("--samples and --max-turns must be positive")

    model = args.model or default_model(args.provider)
    cats = [args.category] if args.category else categories(args.split)

    print(f"P2 Agentic Sweep: {len(cats)} categories x {args.samples} samples (max {args.max_turns} turns)")
    print(f"Provider: {args.provider} [{model}]")

    if args.dry_run:
        print("[dry-run] Configuration validated successfully.")
        return 0

    if not credentials_present(args.provider):
        print(f"No credentials found for {args.provider}. Please set ${PRESETS[args.provider].key_env}")
        return 2

    client = LLMClient(provider=args.provider, model=model)
    store = RunStore(args.runs_dir)

    for cat in cats:
        for s in range(args.samples):
            print(f"\nDrawing {cat} (#{s+1}/{args.samples})...")
            run = store.new_run(dict(
                phase="p2",
                category=cat,
                arm=f"p2_{args.feedback}",
                feedback=args.feedback,
                split=args.split,
                provider=args.provider,
                model=model,
                sample=s,
                max_turns=args.max_turns,
            ))

            try:
                session = run_agent_drawing_session(
                    client=client,
                    category=cat,
                    max_turns=args.max_turns,
                    rows=rows,
                    cols=cols,
                    sample_index=s,
                    feedback=args.feedback,
                )
            except Exception as e:
                print(f"  Failed: {e}")
                store.record(run.finish(error=f"{type(e).__name__}: {e}"))
                continue

            final_grid = session.final_grid or session.turns[-1].grid_snapshot
            run.save_grid("drawing", final_grid)

            # Save progression snapshots
            for t in session.turns:
                run.log(
                    "agent_turn", turn=t.turn, raw_response=t.raw_response,
                    actions=t.actions_executed, results=t.results,
                    finished=t.finished,
                )
                t_grid = t.grid_snapshot
                png_bytes = render_to_png_bytes(t_grid)
                (run.dir / f"turn_{t.turn}.png").write_bytes(png_bytes)
                (run.dir / f"turn_{t.turn}.txt").write_text(t_grid.to_text(), encoding="ascii")

            metrics = structural(final_grid, cat)
            store.record(
                run.finish(
                    turns=len(session.turns),
                    total_actions=session.total_actions(),
                    stop_reason=session.stop_reason,
                    bbox=final_grid.bbox(),
                    **metrics.to_dict(),
                )
            )
            print(f"  Finished {cat} in {len(session.turns)} turns ({session.total_actions()} actions).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
