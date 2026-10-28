"""Summarize a P1 sweep: split x arm, with per-cell means.

    python scripts/analyze_p1.py runs/index.jsonl
    python scripts/analyze_p1.py runs/index.jsonl --metric judge_score

Stdlib only -- no pandas dependency for a table this small.  The headline
comparison is the split x arm interaction: if drawing skill is recall rather
than reasoning, held-out drops sharply, and the arm that closes the gap is the
one carrying transferable method.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ARMS = ("zero_shot", "few_shot", "procedural")
SPLITS = ("seen", "held_out")


def load(path: Path, phase: str = "p1") -> tuple[list[dict], list[dict]]:
    """Return (usable rows, failed-call rows).

    A call that never returned is missing data, not a bad drawing. Counting
    those rows inflates n and drags every mean toward whatever the defaults
    happen to be, so they are separated here rather than filtered silently.
    """
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if r.get("phase") == phase]
    return [r for r in rows if not r.get("error")], [r for r in rows if r.get("error")]


def cell_table(rows: list[dict], metric: str) -> dict[tuple[str, str], list[float]]:
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if metric in r and isinstance(r[metric], (int, float, bool)):
            cells[(r.get("split", "?"), r.get("arm", "?"))].append(float(r[metric]))
    return cells


def fmt(values: list[float]) -> str:
    if not values:
        return "     -    "
    if len(values) == 1:
        return f"  {values[0]:.3f}   "
    return f"{stats.mean(values):.3f}+-{stats.stdev(values):.3f}"


def print_table(title: str, cells: dict, arms: list[str], splits: list[str]) -> None:
    print(f"\n{title}")
    print("  " + "split".ljust(10) + "".join(a.ljust(16) for a in arms))
    for split in splits:
        line = "  " + split.ljust(10)
        for arm in arms:
            line += fmt(cells.get((split, arm), [])).ljust(16)
        print(line)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("index", type=Path, nargs="?", default=Path("runs/index.jsonl"))
    ap.add_argument("--model", default=None, help="restrict to one model")
    ap.add_argument(
        "--metric",
        default=None,
        help="metric to tabulate; default reports every numeric one found",
    )
    args = ap.parse_args(argv)

    if not args.index.exists():
        print(f"no index at {args.index} -- run the sweep first")
        return 1

    rows, failed = load(args.index)
    if not rows:
        print("no usable p1 rows in index")
        return 1

    models = sorted({r.get("model", "?") for r in rows})
    if args.model:
        rows = [r for r in rows if r.get("model") == args.model]
        models = [args.model]
    elif len(models) > 1:
        # Pooling two models into one mean compares nothing. Say so loudly
        # and tabulate them separately.
        print(f"NOTE: {len(models)} models in this index -- reported "
              f"separately, never pooled:")
        for m in models:
            print(f"   {m}  ({sum(1 for r in rows if r.get('model') == m)} runs)")
        print()

    if failed:
        print(f"{len(failed)} call(s) failed and are excluded (missing data, "
              f"not bad drawings):")
        for model in sorted({r.get("model", "?") for r in failed}):
            n = sum(1 for r in failed if r.get("model") == model)
            print(f"   {model}: {n}")
        print()

    for model in models:
        subset = [r for r in rows if r.get("model") == model]
        if not subset:
            continue
        if len(models) > 1:
            print(f"{'=' * 60}\nmodel: {model}\n{'=' * 60}")
        report_model(subset, args.metric)
    return 0


def report_model(rows: list[dict], metric: str | None) -> None:
    arms = [a for a in ARMS if any(r.get("arm") == a for r in rows)]
    splits = [s for s in SPLITS if any(r.get("split") == s for r in rows)]

    print(f"{len(rows)} runs  |  {len(arms)} arms  |  {len(splits)} splits")
    print(f"total cost: ${sum(r.get('cost_usd', 0) or 0 for r in rows):.4f}")

    counts = {(s, a): sum(1 for r in rows if r.get("split") == s and r.get("arm") == a)
              for s in splits for a in arms}
    print_table("n per cell", {k: [v] for k, v in counts.items()}, arms, splits)

    if metric:
        metrics = [metric]
    else:
        skip = {"sample", "cost_usd", "elapsed_s"}
        metrics = sorted(
            {
                k
                for r in rows
                for k, v in r.items()
                if isinstance(v, (int, float, bool)) and k not in skip
            }
        )

    for metric in metrics:
        cells = cell_table(rows, metric)
        if cells:
            print_table(f"{metric}  (mean+-sd)", cells, arms, splits)

    # Failure modes are worth naming individually, not just averaging.
    bad = [r for r in rows if r.get("spells_name") or r.get("degenerate")
           or r.get("empty_response") or not r.get("parse_clean", True)]
    if bad:
        print(f"\n{len(bad)} run(s) need a look:")
        for r in bad[:20]:
            why = []
            if r.get("empty_response"):
                why.append("empty_response")
            elif not r.get("parse_clean", True):
                why += r.get("parse_issues", ["malformed"])
            if r.get("spells_name"):
                why.append("spells_name")
            if r.get("degenerate"):
                why.append("degenerate")
            print(f"  {r.get('category','?'):16s} {r.get('arm','?'):11s} "
                  f"#{r.get('sample','?')}  {','.join(why)}")


if __name__ == "__main__":
    raise SystemExit(main())
