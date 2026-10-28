"""P5 Experiment: Semantic Image-to-ASCII Conversion.

This is *not* naive pixel-to-ASCII (no jp2a/AALib-style character mapping).
Instead, a multimodal LLM is asked to understand the *semantic content* of the
input photograph and produce ASCII art that captures the essential silhouette
and structural features of the primary subject.

Pipeline:
  1. Load the input image with PIL; resize to ≤ 512 px on the longest side.
  2. Encode as PNG bytes.
  3. Send to the LLM with a two-part system prompt:
       (a) Semantic perception instruction: focus on the primary subject,
           ignore background clutter, identify key silhouette/structure.
       (b) Standard P1 canvas contract: 24×48, printable ASCII, fenced block.
  4. Parse the model's output as an AsciiGrid.
  5. Save the grid text and its rendered PNG to the run directory.

Usage::

    python -m asciigent.experiments.p5_image2ascii --image /path/to/photo.jpg
    python -m asciigent.experiments.p5_image2ascii \\
        --image /path/to/photo.jpg --provider gemini --runs-dir runs
    python -m asciigent.experiments.p5_image2ascii --dry-run \\
        --image /path/to/photo.png
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from asciigent.grid import DEFAULT_COLS, DEFAULT_ROWS, AsciiGrid
from asciigent.llm import DEFAULT_PROVIDER, LLMClient
from asciigent.providers import PRESETS, credentials_present, default_model
from asciigent.render import render_to_png_bytes, save_png

# Maximum pixel size (longest side) before we resize
MAX_IMAGE_PX = 512

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SEMANTIC_PERCEPTION = (
    "You are converting a photograph into ASCII art. "
    "Focus only on the primary subject - ignore background clutter, textures, "
    "and noise. Identify the key silhouette and structural features "
    "(outlines, facial features, major edges). "
    "Generate ASCII art that captures the semantic essence of the subject."
)

_CANVAS_CONTRACT = (
    "Canvas rules:\n"
    f"- The canvas is exactly {DEFAULT_ROWS} rows tall and {DEFAULT_COLS} columns wide.\n"
    "- Use only printable ASCII (space through ~). No Unicode, no box-drawing "
    "characters, no emoji, no tabs.\n"
    "- Use spaces for blank areas. Leading spaces are meaningful.\n"
    "- Output the drawing inside a single fenced code block and nothing else after it.\n"
    "- Draw the shape of the thing. Do not spell its name."
)

IMAGE2ASCII_SYSTEM = f"{_SEMANTIC_PERCEPTION}\n\n{_CANVAS_CONTRACT}"


def _load_and_resize_image(image_path: str) -> bytes:
    """Load an image, resize to ≤ MAX_IMAGE_PX on longest side, return PNG bytes."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > MAX_IMAGE_PX:
        scale = MAX_IMAGE_PX / longest
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _user_prompt() -> str:
    return (
        f"Convert the photograph above into ASCII art on the "
        f"{DEFAULT_ROWS}×{DEFAULT_COLS} canvas. "
        "Output only the fenced code block containing the drawing."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True,
                    help="Path to the input image (JPEG, PNG, etc.).")
    ap.add_argument("--provider", default=DEFAULT_PROVIDER, choices=sorted(PRESETS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print configuration and exit without calling the API.")
    args = ap.parse_args(argv)

    model = args.model or default_model(args.provider)
    runs_dir = Path(args.runs_dir)
    image_path = Path(args.image)

    print("P5 Semantic Image-to-ASCII")
    print(f"  image    : {image_path}")
    print(f"  provider : {args.provider} [{model}]")
    print(f"  runs-dir : {runs_dir}")

    if args.dry_run:
        print("\n[dry-run] Checking image path …")
        if not image_path.exists():
            print(f"[dry-run] Note: image path does not exist: {image_path}  (OK in dry-run)")
        else:
            from PIL import Image
            img = Image.open(str(image_path))
            print(f"[dry-run] Image size: {img.size[0]}×{img.size[1]} px")
        print(f"[dry-run] System prompt preview:\n{IMAGE2ASCII_SYSTEM[:400]}")
        print("\n[dry-run] Configuration validated successfully.")
        return 0

    if not credentials_present(args.provider):
        cfg = PRESETS[args.provider]
        print(f"No credentials found for {args.provider}. Please set ${cfg.key_env}")
        return 2

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 1

    # -- Load and encode image ------------------------------------------------
    print("\nLoading and encoding image …")
    png_bytes = _load_and_resize_image(str(image_path))
    print(f"Encoded image: {len(png_bytes):,} bytes")

    # -- Call LLM -------------------------------------------------------------
    client = LLMClient(provider=args.provider, model=model)
    resp = client.complete(
        prompt=_user_prompt(),
        images=[png_bytes],
        system=IMAGE2ASCII_SYSTEM,
    )

    print("\n--- LLM response (first 500 chars) ---")
    print(resp.text[:500])

    # -- Parse output ---------------------------------------------------------
    grid, report = AsciiGrid.from_text(resp.text, rows=DEFAULT_ROWS, cols=DEFAULT_COLS)
    print(f"\nParsed grid: {grid.rows}×{grid.cols}, ink_ratio={grid.ink_ratio():.3f}")
    if not report.clean:
        print(f"  Parse issues: {report.issues}")

    # -- Save -----------------------------------------------------------------
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_subdir = runs_dir / "p5_image2ascii"
    run_subdir.mkdir(parents=True, exist_ok=True)

    (run_subdir / "output.txt").write_text(grid.to_text(), encoding="ascii")
    save_png(grid, str(run_subdir / "output.png"))
    (run_subdir / "response.txt").write_text(resp.text, encoding="utf-8")

    print(f"\nSaved to: {run_subdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
