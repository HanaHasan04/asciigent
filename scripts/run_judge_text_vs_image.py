"""Demonstrate judge text-vs-image comparison (P3 extension).

Judges a single ASCII drawing TWICE:

1. **Image judge** – the standard pipeline from ``judge.py``: the drawing is
   rendered to a PNG and the judge sees *only* the bitmap.
2. **Text judge** – the same drawing is passed as a raw character grid (the
   ``numbered_text()`` representation); no image is sent.

The two verdicts are printed side-by-side so you can see whether the text judge
reads individual characters rather than perceiving shapes.

The "text-reading confound" (Wang et al.): if the text judge correctly
identifies the subject *and* the drawing spells the name in characters, that is
evidence the judge is exploiting textual cues rather than visual structure.

Usage::

    python scripts/run_judge_text_vs_image.py --drawing path/to/cat.txt \\
        --category cat
    python scripts/run_judge_text_vs_image.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asciigent.grid import DEFAULT_COLS, DEFAULT_ROWS, AsciiGrid
from asciigent.judge import Verdict, judge_image, matches, parse_rating
from asciigent.llm import DEFAULT_PROVIDER, LLMClient
from asciigent.metrics import structural
from asciigent.providers import PRESETS, credentials_present, default_model
from asciigent.render import render_to_png_bytes

# ---------------------------------------------------------------------------
# Built-in fallback drawing (same cat face used in P4)
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
# Text-judge system prompt
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_TEXT = (
    "You are shown an ASCII art drawing as a character grid with row/column numbers. "
    "Judge it as a visual picture: what shapes and lines depict. "
    "Do not read individual characters as words."
)

_IDENTIFY_USER = (
    "What single thing does this picture depict?\n\n"
    "Answer with a short noun phrase of one to three words, lowercase, and nothing "
    "else. If it does not depict any recognizable thing, answer exactly: unclear"
)

_RATE_USER_TMPL = (
    "How recognizable is this picture as a {category}?\n\n"
    "Use this scale:\n"
    "5 - unmistakably a {category}\n"
    "4 - clearly a {category}, with flaws\n"
    "3 - plausibly a {category}, but ambiguous\n"
    "2 - barely suggests a {category}\n"
    "1 - not a {category} at all, or nothing recognizable\n\n"
    "Answer with the single digit and nothing else."
)

JUDGE_MAX_TOKENS = 2048


def _judge_text(
    client: LLMClient,
    grid: AsciiGrid,
    category: str,
) -> Verdict:
    """Judge the drawing using its text representation (no image)."""
    text_prompt = grid.numbered_text()

    ident = client.complete(
        prompt=f"{text_prompt}\n\n{_IDENTIFY_USER}",
        system=JUDGE_SYSTEM_TEXT,
        max_tokens=JUDGE_MAX_TOKENS,
        effort="low",
    )
    answer = ident.text.strip().splitlines()[0] if ident.text.strip() else ""

    rate_prompt = f"{text_prompt}\n\n{_RATE_USER_TMPL.format(category=category)}"
    rated = client.complete(
        prompt=rate_prompt,
        system=JUDGE_SYSTEM_TEXT,
        max_tokens=JUDGE_MAX_TOKENS,
        effort="low",
    )

    return Verdict(
        variant="text",
        identified=answer,
        correct=matches(answer, category),
        rating=parse_rating(rated.text),
        raw_rating=rated.text.strip()[:40],
        failed=ident.empty or rated.empty,
    )


def _load_drawing(path: str | None) -> AsciiGrid:
    if path:
        text = Path(path).read_text(encoding="ascii", errors="replace")
        grid, _ = AsciiGrid.from_text(text, rows=DEFAULT_ROWS, cols=DEFAULT_COLS, extract=False)
        return grid
    grid, _ = AsciiGrid.from_text(
        _DEFAULT_DRAWING, rows=DEFAULT_ROWS, cols=DEFAULT_COLS, extract=False
    )
    return grid


def _fmt_verdict(v: Verdict) -> str:
    rating = str(v.rating) if v.rating is not None else "?"
    correct = "✓" if v.correct else "✗"
    failed = " [FAILED]" if v.failed else ""
    return f"identified={v.identified!r:20s}  rating={rating}  correct={correct}{failed}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drawing", default=None,
                    help="Path to a .txt ASCII drawing. Uses built-in cat face if omitted.")
    ap.add_argument("--category", default="cat",
                    help="True category label for identification accuracy (default: cat).")
    ap.add_argument("--provider", default=DEFAULT_PROVIDER, choices=sorted(PRESETS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print configuration and exit without calling the API.")
    args = ap.parse_args(argv)

    model = args.model or default_model(args.provider)

    print("Judge Text-vs-Image Comparison")
    print(f"  drawing  : {args.drawing or '(built-in cat face)'}")
    print(f"  category : {args.category}")
    print(f"  provider : {args.provider} [{model}]")

    if args.dry_run:
        print("\n[dry-run] Loading drawing …")
        grid = _load_drawing(args.drawing)
        print(f"[dry-run] Canvas: {grid.rows}×{grid.cols}, ink_ratio={grid.ink_ratio():.3f}")
        m = structural(grid, args.category)
        print(f"[dry-run] Structural metrics: {m.to_dict()}")
        print(f"[dry-run] Text judge system:\n  {JUDGE_SYSTEM_TEXT}")
        print("\n[dry-run] Configuration validated successfully.")
        return 0

    if not credentials_present(args.provider):
        cfg = PRESETS[args.provider]
        print(f"No credentials found for {args.provider}. Please set ${cfg.key_env}")
        return 2

    grid = _load_drawing(args.drawing)
    png = render_to_png_bytes(grid)
    m = structural(grid, args.category)
    spells_name = m.spells_name

    client = LLMClient(provider=args.provider, model=model)

    print("\nRunning image judge …")
    img_verdict = judge_image(client, png, args.category, variant="real")

    print("Running text judge …")
    txt_verdict = _judge_text(client, grid, args.category)

    # -- Print comparison table -----------------------------------------------
    print("\n" + "=" * 72)
    print(f"  Category   : {args.category}")
    print(f"  Spells name: {spells_name}")
    print("-" * 72)
    print(f"  Image judge: {_fmt_verdict(img_verdict)}")
    print(f"  Text  judge: {_fmt_verdict(txt_verdict)}")
    print("=" * 72)

    if spells_name and txt_verdict.correct and not img_verdict.correct:
        print(
            "\n⚠  TEXT-READING CONFOUND DETECTED: text judge correctly identified "
            "the subject, but the image judge did not — and the drawing spells "
            "the name. The text judge is likely reading characters, not shapes."
        )
    elif txt_verdict.correct and img_verdict.correct:
        print("\n✓ Both judges agree. No confound detected (both correct).")
    elif not txt_verdict.correct and not img_verdict.correct:
        print("\n✗ Neither judge identified the drawing correctly.")
    else:
        print("\n→ Judges disagree. Further analysis needed.")

    print(f"\nLLM calls: {client.n_calls}, cache hits: {client.n_cache_hits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
