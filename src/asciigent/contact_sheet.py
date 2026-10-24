"""Tile many renders into one labelled sheet.

Two uses. During the pilot it is how you look at 42 drawings in one glance
instead of opening 42 files -- the fastest way to catch a systematic format or
scale problem before spending on a full sweep. Later it is the raw material for
the P3 human-rating study.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .render import RenderStyle, _load_font

LABEL_H = 22
GAP = 10


def _label_font(size: int = 13):
    font, _ = _load_font(RenderStyle(font_size=size))
    return font


def contact_sheet(
    tiles: list[tuple[str, Image.Image]],
    ncols: int = 6,
    bg: tuple[int, int, int] = (245, 245, 245),
) -> Image.Image:
    """Arrange ``(label, image)`` pairs into a labelled grid."""
    if not tiles:
        raise ValueError("no tiles to lay out")

    cell_w = max(img.width for _, img in tiles)
    cell_h = max(img.height for _, img in tiles) + LABEL_H
    ncols = min(ncols, len(tiles))
    nrows = (len(tiles) + ncols - 1) // ncols

    sheet = Image.new(
        "RGB",
        (ncols * cell_w + (ncols + 1) * GAP, nrows * cell_h + (nrows + 1) * GAP),
        bg,
    )
    draw = ImageDraw.Draw(sheet)
    font = _label_font()

    for i, (label, img) in enumerate(tiles):
        r, c = divmod(i, ncols)
        x = GAP + c * (cell_w + GAP)
        y = GAP + r * (cell_h + GAP)
        draw.text((x, y + 4), label[:64], font=font, fill=(40, 40, 40))
        sheet.paste(img, (x, y + LABEL_H))
    return sheet


def sheet_from_runs(
    run_dirs: list[Path],
    image_name: str = "drawing.png",
    label_keys: tuple[str, ...] = ("category", "arm", "sample"),
    ncols: int = 6,
    scale: float = 0.45,
) -> Image.Image:
    """Build a sheet from run directories, labelling each by its config.

    Tiles are ordered by ``label_keys``, not by run id, so the same category
    sits on the same row across arms and the comparison reads left-to-right.
    ``scale`` shrinks each render -- a full canvas at reading size makes a
    42-cell sheet unusably large, and the point of the sheet is the overview.
    """
    import json

    entries = []
    for d in run_dirs:
        img_path = d / image_name
        cfg_path = d / "config.json"
        if not img_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        key = tuple(str(cfg.get(k, "")) for k in label_keys)
        entries.append((key, " / ".join(key), img_path))

    tiles = []
    for _, label, img_path in sorted(entries):
        img = Image.open(img_path)
        if scale != 1.0:
            img = img.resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                Image.LANCZOS,
            )
        tiles.append((label, img))
    return contact_sheet(tiles, ncols=ncols)
