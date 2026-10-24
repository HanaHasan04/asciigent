"""Verify a provider works, in three calls, before spending quota on a sweep.

    python scripts/check_provider.py
    python scripts/check_provider.py --provider openrouter

Three checks, cheapest first:

1. **Auth and model name.** A plain text call. Catches a bad key or a retired
   model id immediately, which is the usual first failure.
2. **Vision.** Sends an image and asks what is in it. Catches a text-only
   model being used for a task that needs eyes.
3. **ASCII perception.** Renders a *human-made* ASCII cat and asks the model
   to name it, seeing only the picture. This is the prerequisite P2 rests on:
   if a model cannot recognize a good ASCII drawing from its render, then
   asking it to critique its own rendered drawing is meaningless, and the
   whole self-perception loop would measure nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asciigent.grid import AsciiGrid  # noqa: E402
from asciigent.llm import DEFAULT_PROVIDER, LLMClient  # noqa: E402
from asciigent.providers import PRESETS, credentials_present, default_model  # noqa: E402
from asciigent.render import render_to_png_bytes  # noqa: E402

# A clear, human-made drawing. If the model misses this, it will miss anything
# a model produces.
REFERENCE_CAT = r"""
 /\_/\
( o.o )
 > ^ <
 (")_(")
""".strip("\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default=DEFAULT_PROVIDER, choices=sorted(PRESETS))
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)

    cfg = PRESETS[args.provider]
    model = args.model or default_model(args.provider)
    print(f"provider : {args.provider}")
    print(f"model    : {model}")
    print(f"key var  : {cfg.key_env or '(none needed)'}")

    if not credentials_present(args.provider):
        print(f"\nFAIL: {cfg.key_env} is not set.")
        print("  Put it in a .env file at the repo root:")
        print(f"      {cfg.key_env}=your-key-here")
        print("  (.env is git-ignored. Copy .env.example to start.)")
        return 2
    print("key      : found\n")

    client = LLMClient(provider=args.provider, model=model)

    def attempt(label: str, hint: str, **kw):
        """Run one call. An empty body is a failure, not a pass.

        A thinking model that spends its whole budget on reasoning returns
        HTTP 200 with no content -- so checking only for an exception reports
        success on a call that produced nothing.
        """
        try:
            resp = client.complete(effort="low", **kw)
        except Exception as exc:  # noqa: BLE001 - report any failure plainly
            print(f"{label} FAIL\n   {type(exc).__name__}: {str(exc)[:300]}")
            print(f"   {hint}")
            return None
        if resp.empty:
            print(f"{label} FAIL -> empty response"
                  f"{' (truncated)' if resp.truncated else ''}")
            print("   The model returned no text. Raise max_tokens or lower "
                  "effort:\n   on thinking models the reasoning is billed "
                  "against the output cap.")
            return None
        print(f"{label} OK   -> {resp.text.strip()[:50]!r}")
        return resp

    # 1 -- auth and model id
    if attempt("1. text call  ", "Bad key, or a retired model id.",
               prompt="Reply with exactly: OK", max_tokens=2048) is None:
        return 1

    # 2 -- vision at all
    solid = AsciiGrid.from_text("\n".join(["#" * 20] * 6), rows=8, cols=24,
                                extract=False)[0]
    if attempt("2. vision     ", "This model may be text-only. P2 needs eyes.",
               prompt="Is there a solid filled rectangle in this image? "
                      "Answer yes or no.",
               images=[render_to_png_bytes(solid)], max_tokens=2048) is None:
        return 1

    # 3 -- can it read ASCII art as a picture?
    cat, _ = AsciiGrid.from_text(REFERENCE_CAT, rows=8, cols=24, extract=False)
    percept = attempt(
        "3. ascii sight",
        "Could not test ASCII perception.",
        prompt="What single thing does this picture depict? "
               "Answer with one to three lowercase words and nothing else.",
        images=[render_to_png_bytes(cat)], max_tokens=2048,
    )
    if percept is None:
        return 1
    answer = percept.text.strip().lower()
    ok = any(w in answer for w in ("cat", "kitten", "kitty", "feline"))
    if not ok:
        print(f"   ^ WEAK: expected a cat, got {answer[:60]!r}")

    print(f"\n{client.summary()}")
    if not ok:
        print(
            "\nNote: the model did not recognize a clear, human-made ASCII cat\n"
            "from its render. P1 can still run, but P2's premise -- that a model\n"
            "can perceive its own rendered drawing -- looks shaky on this model.\n"
            "Worth trying another before building the loop on it."
        )
    else:
        print("\nAll three checks passed. Ready to run the P1 pilot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
