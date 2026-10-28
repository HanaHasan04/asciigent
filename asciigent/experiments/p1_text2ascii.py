"""P1 sweep: category split x prompting arm.

Usage (from the repo root):

    python -m asciigent.experiments.p1_text2ascii --dry-run
    python -m asciigent.experiments.p1_text2ascii --split seen --samples 2
    python -m asciigent.experiments.p1_text2ascii --provider openrouter

Runs on free-tier providers only; a metered provider is refused outright.
``--dry-run`` prints what would be sent and calls nothing.  Re-running a sweep
re-calls only the cells that actually changed, because responses are cached --
which matters on a free tier, where the daily request quota is the budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python src/asciigent/experiments/p1_...py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from asciigent.contact_sheet import sheet_from_runs
from asciigent.grid import AsciiGrid
from asciigent.llm import DEFAULT_PROVIDER, LLMClient
from asciigent.providers import (
    PRESETS,
    DailyQuotaExhausted,
    credentials_present,
    default_model,
)
from asciigent.metrics import structural
from asciigent.prompts.text2ascii import ARMS, build, system_prompt
from asciigent.store import RunStore

DATA = Path(__file__).resolve().parents[2] / "data"


def categories(split: str) -> list[str]:
    meta = json.loads((DATA / "categories.json").read_text(encoding="utf-8"))
    if split == "all":
        return meta["seen"] + meta["held_out"]
    return meta[split]


def main(argv: list[str] | None = None) -> int:
    meta = json.loads((DATA / "categories.json").read_text(encoding="utf-8"))
    rows, cols = meta["canvas"]["rows"], meta["canvas"]["cols"]

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="all", choices=["seen", "held_out", "all"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument(
        "--provider", default=DEFAULT_PROVIDER, choices=sorted(PRESETS),
        help="free by default; metered providers are refused",
    )
    ap.add_argument("--model", default=None, help="defaults to the provider's model")
    ap.add_argument("--samples", type=int, default=1, help="draws per cell")
    ap.add_argument("--effort", default=None, help="low|medium|high|xhigh|max")
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    model = args.model or default_model(args.provider)
    cats = categories(args.split)
    system = system_prompt(rows, cols)
    cells = [(c, a, s) for c in cats for a in args.arms for s in range(args.samples)]

    cfg = PRESETS[args.provider]
    pace = f", ~{cfg.rpm:g} req/min" if cfg.rpm else ""
    eta = f" (~{len(cells) / cfg.rpm:.0f} min)" if cfg.rpm else ""
    print(f"{len(cells)} cells: {len(cats)} categories x {len(args.arms)} arms "
          f"x {args.samples} samples")
    print(f"provider: {args.provider} [{model}]"
          f"{' FREE' if not cfg.metered else ' METERED'}{pace}{eta}")

    if args.dry_run:
        cat, arm = cats[0], args.arms[0]
        print(f"\n--- SYSTEM ---\n{system}\n\n--- USER ({arm} / {cat}) ---")
        print(build(arm, cat, rows=rows, cols=cols))
        return 0

    if not credentials_present(args.provider):
        print(f"\nno credentials: set ${PRESETS[args.provider].key_env} "
              f"(free key from the provider's console), then re-run.")
        return 2

    client = LLMClient(provider=args.provider, model=model)
    store = RunStore(args.runs_dir)
    n_malformed = 0
    run_dirs = []
    failures: list[tuple] = []
    quota_hit = False

    for cat, arm, sample in cells:
        prompt = build(arm, cat, rows=rows, cols=cols)
        run = store.new_run(
            {
                "phase": "p1",
                "category": cat,
                "split": "seen" if cat in meta["seen"] else "held_out",
                "arm": arm,
                "sample": sample,
                "model": model,
                "effort": args.effort,
            }
        )
        # Free tiers return 503 under load and 429 at the quota edge. One bad
        # cell must not discard the cells already drawn -- record it, carry on,
        # and let a re-run fill the gap from cache plus the missing calls.
        try:
            resp = client.complete(
                prompt,
                system=system,
                model=model,
                max_tokens=args.max_tokens,
                effort=args.effort,
                cache_system=True,
                sample_index=sample,
            )
        except DailyQuotaExhausted as exc:
            # Waiting will not help today. Stop now: the cells already
            # drawn are saved and cached, so tomorrow's re-run resumes.
            store.record(run.finish(error='DailyQuotaExhausted'))
            print(f"\n  {cat} / {arm} #{sample}: DAILY QUOTA EXHAUSTED")
            print(f"  {str(exc)[:200]}")
            quota_hit = True
            break
        except Exception as exc:  # noqa: BLE001 - any provider failure
            failures.append((cat, arm, sample, f"{type(exc).__name__}: {exc}"))
            store.record(
                run.finish(error=f"{type(exc).__name__}", parse_clean=False)
            )
            print(f"  {cat:16s} {arm:11s} #{sample}  FAILED "
                  f"({type(exc).__name__})")
            continue

        grid, report = AsciiGrid.from_text(resp.text, rows=rows, cols=cols)
        metrics = structural(grid, cat)
        n_malformed += not report.clean

        (run.dir / "response.txt").write_text(resp.text, encoding="utf-8")
        run.save_grid("drawing", grid)
        run.log("draft", **resp.to_dict(), parse=report.to_dict())
        store.record(
            run.finish(
                parse_clean=report.clean,
                parse_issues=report.issues,
                bbox=grid.bbox(),
                digest=grid.digest(),
                cached=resp.cached,
                cost_usd=round(resp.cost_usd, 5),
                truncated=resp.truncated,
                empty_response=resp.empty,
                **metrics.to_dict(),
            )
        )
        run_dirs.append(run.dir)

        flags = list(report.issues)
        if metrics.spells_name:
            flags.append("SPELLS-NAME")
        # An empty reply also yields a blank canvas. Reporting that as
        # "the model drew nothing" would blame the model for what is really
        # a truncated call, so name it separately.
        if resp.empty:
            flags.append("EMPTY-RESPONSE")
        elif metrics.degenerate:
            flags.append("DEGENERATE")
        if resp.truncated:
            flags.append("truncated")
        suffix = f"  [{','.join(flags)}]" if flags else ""
        print(
            f"  {cat:16s} {arm:11s} #{sample}  "
            f"ink={metrics.ink_ratio:.3f} cov={metrics.coverage:.2f}{suffix}"
        )

    sheet_path = Path(args.runs_dir) / "p1_contact_sheet.png"
    if run_dirs:
        sheet_from_runs(run_dirs, ncols=len(args.arms) or 3).save(sheet_path)

    print(f"\n{json.dumps(client.summary(), indent=2)}")
    print(f"malformed: {n_malformed}/{len(cells)}")
    if failures:
        print(f"failed: {len(failures)}/{len(cells)} -- re-run to retry "
              f"just these; finished cells come from cache")
        for cat, arm, sample, why in failures[:10]:
            print(f"    {cat} / {arm} #{sample}: {why[:90]}")
    if quota_hit:
        print(
            "\nStopped early: the daily free-tier allowance for this model is "
            "gone.\nRe-run tomorrow (finished cells load from cache), or pass "
            "--model with a\ndifferent free model -- the quota is per model, "
            "so each has its own allowance."
        )
    print(f"index: {store.index}")
    print(f"contact sheet: {sheet_path}  <- look at this before trusting anything")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
