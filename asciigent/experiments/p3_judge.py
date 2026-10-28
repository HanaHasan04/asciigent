"""Judge an existing sweep.

Generation and judging are separate runs on purpose: the drawings are fixed
data, and the judge is a variable we expect to change several times while P3
calibrates it. Re-judging must never mean re-drawing.

    python -m asciigent.experiments.p3_judge --runs-dir runs
    python -m asciigent.experiments.p3_judge --runs-dir runs --controls cells ink
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from asciigent.grid import DEFAULT_COLS, DEFAULT_ROWS, AsciiGrid
from asciigent.judge import judge_drawing
from asciigent.llm import DEFAULT_PROVIDER, LLMClient
from asciigent.providers import PRESETS, credentials_present, default_model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--provider", default=DEFAULT_PROVIDER, choices=sorted(PRESETS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--controls", nargs="*", default=["cells"], choices=["cells", "ink"])
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.runs_dir)
    index_path = root / "index.jsonl"
    if not index_path.exists():
        print(f"no sweep at {index_path} -- generate first")
        return 1

    rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if r.get("run_id") and (root / r["run_id"] / "drawing.txt").exists()]
    if args.limit:
        rows = rows[: args.limit]

    model = args.model or default_model(args.provider)
    per_drawing = 2 * (1 + len(args.controls))  # identify + rate, real + controls
    total = len(rows) * per_drawing
    cfg = PRESETS[args.provider]
    eta = f"  (~{total / cfg.rpm:.0f} min at {cfg.rpm:g} rpm)" if cfg.rpm else ""
    print(f"{len(rows)} drawings x {per_drawing} calls = {total} calls "
          f"[{args.provider}/{model}]{eta}")

    if args.dry_run:
        return 0
    if not credentials_present(args.provider):
        print(f"no credentials: set ${cfg.key_env}")
        return 2

    client = LLMClient(provider=args.provider, model=model)
    out_path = root / "judgements.jsonl"
    scores: list[int] = []
    n_failed = 0
    gaps: list[float] = []

    with out_path.open("a", encoding="utf-8") as fh:
        for row in rows:
            run_dir = root / row["run_id"]
            text = (run_dir / "drawing.txt").read_text(encoding="ascii")
            grid, _ = AsciiGrid.from_text(
                text, rows=DEFAULT_ROWS, cols=DEFAULT_COLS, extract=False
            )
            verdict = judge_drawing(
                client, grid, row["category"], controls=tuple(args.controls)
            )
            record = {"run_id": row["run_id"], "arm": row.get("arm"),
                      "split": row.get("split"), **verdict.to_dict()}
            fh.write(json.dumps(record) + "\n")
            fh.flush()  # free tiers die mid-sweep; never lose finished work

            if verdict.real and verdict.real.failed:
                n_failed += 1
            if verdict.real and verdict.real.rating:
                scores.append(verdict.real.rating)
            gap = verdict.control_gap()
            if gap is not None:
                gaps.append(gap)

            mark = "OK " if verdict.real and verdict.real.correct else "   "
            print(f"  {mark}{row['category']:16s} {row.get('arm',''):11s} "
                  f"score={verdict.real.rating if verdict.real else '?'} "
                  f"said={verdict.real.identified if verdict.real else ''!r}")

    print(f"\n{json.dumps(client.summary(), indent=2)}")
    if n_failed:
        print(f"{n_failed} judge call(s) returned nothing -- excluded, not scored 1")
    if scores:
        print(f"mean score: {stats.mean(scores):.2f}  (n={len(scores)})")
    if gaps:
        mean_gap = stats.mean(gaps)
        print(f"mean control gap: {mean_gap:+.2f}")
        if mean_gap < 1.0:
            print("  WARNING: judge scores scrambled images nearly as highly as "
                  "real ones.\n  It is not measuring shape -- treat every score "
                  "as void until this is fixed.")
    print(f"judgements: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
