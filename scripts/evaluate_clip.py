"""Evaluate rendered ASCII drawings using CLIP embedding similarity.

Usage:
    python scripts/evaluate_clip.py runs/index.jsonl
    python scripts/evaluate_clip.py --drawing data/exemplars/fish.txt --category fish
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asciigent.grid import AsciiGrid, DEFAULT_ROWS, DEFAULT_COLS
from asciigent.metrics_vision import evaluate_vision_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("index_or_file", nargs="?", default="runs/index.jsonl")
    ap.add_argument("--drawing", default=None, help="Path to single ASCII text file")
    ap.add_argument("--category", default=None, help="Target category name")
    args = ap.parse_args()

    if args.drawing:
        category = args.category or Path(args.drawing).stem
        text = Path(args.drawing).read_text(encoding="utf-8")
        grid, _ = AsciiGrid.from_text(text, extract=False)
        metrics = evaluate_vision_metrics(grid, category)
        print(f"\nDrawing: {args.drawing} (Category: {category})")
        for k, v in metrics.items():
            print(f"  {k:26s}: {v:.4f}")
        return 0

    index_path = Path(args.index_or_file)
    if not index_path.exists():
        print(f"File not found: {index_path}")
        return 1

    lines = [json.loads(l) for l in index_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    valid_runs = [r for r in lines if r.get("run_id") and (index_path.parent / r["run_id"] / "drawing.txt").exists()]

    if not valid_runs:
        print(f"No run directories found with drawing.txt under {index_path.parent}")
        return 0

    print(f"\nEvaluating {len(valid_runs)} drawings with CLIP ViT-B/32...\n")
    print(f"{'Run ID':24s} {'Cat':10s} {'Arm':12s} {'CLIP Real':10s} {'CLIP Scramble':14s} {'Control Gap':12s}")
    print("-" * 88)

    for r in valid_runs:
        run_id = r["run_id"]
        cat = r["category"]
        arm = r.get("arm", "unknown")
        txt_path = index_path.parent / run_id / "drawing.txt"
        text = txt_path.read_text(encoding="utf-8")
        grid, _ = AsciiGrid.from_text(text, extract=False)

        metrics = evaluate_vision_metrics(grid, cat)
        scramble_max = max(metrics["clip_score_scramble_cells"], metrics["clip_score_scramble_ink"])
        gap = metrics["clip_control_gap"]

        print(f"{run_id:24s} {cat:10s} {arm:12s} {metrics['clip_score_real']:10.4f} {scramble_max:14.4f} {gap:12.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
