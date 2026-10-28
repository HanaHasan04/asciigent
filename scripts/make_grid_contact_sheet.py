"""Generate a clean, publication-ready 4x3 contact sheet for P1 baseline.

Columns: Cat, Dog, Bear, Horse
Rows: Zero-Shot, Few-Shot, Procedural
"""

from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont, ImageEnhance


def build_p1_grid():
    root = Path(__file__).resolve().parents[1]
    runs_dir = root / "runs"
    runs_index = runs_dir / "index.jsonl"
    entries = [json.loads(line) for line in runs_index.read_text(encoding="utf-8").splitlines() if line.strip()]

    p1_runs = {}
    for e in entries:
        if e.get("model") == "gemini-3-flash-preview" and not e.get("truncated") and e.get("arm") in ("zero_shot", "few_shot", "procedural"):
            p1_runs[(e["category"], e["arm"])] = runs_dir / e["run_id"]

    categories = ["cat", "dog", "bear", "horse"]
    arms = ["zero_shot", "few_shot", "procedural"]
    arm_labels = ["Zero-Shot", "Few-Shot", "Procedural"]
    cat_labels = ["Cat ('cat')", "Dog ('dog')", "Bear ('bear')", "Horse ('horse')"]

    images = {}
    for c in categories:
        for a in arms:
            rd = p1_runs.get((c, a))
            if rd and (rd / "drawing.png").exists():
                im = Image.open(rd / "drawing.png").convert("RGB")
                images[(c, a)] = im
            else:
                raise RuntimeError(f"Missing drawing for ({c}, {a})")

    scale = 0.75
    sample_img = next(iter(images.values()))
    cell_w = int(sample_img.width * scale)
    cell_h = int(sample_img.height * scale)

    header_top = 34
    row_header_left = 95
    gap = 10
    margin = 10

    total_w = row_header_left + 4 * cell_w + 3 * gap + 2 * margin
    total_h = header_top + 3 * cell_h + 2 * gap + 2 * margin

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font_header = ImageFont.truetype("arialbd.ttf", 18)
        font_row = ImageFont.truetype("arialbd.ttf", 16)
    except Exception:
        font_header = ImageFont.load_default()
        font_row = ImageFont.load_default()

    for j, label in enumerate(cat_labels):
        col_x = margin + row_header_left + j * (cell_w + gap)
        bbox = draw.textbbox((0, 0), label, font=font_header)
        tw = bbox[2] - bbox[0]
        tx = col_x + (cell_w - tw) // 2
        draw.text((tx, margin + 6), label, font=font_header, fill=(15, 15, 15))

    for i, (arm_key, arm_lbl) in enumerate(zip(arms, arm_labels)):
        row_y = margin + header_top + i * (cell_h + gap)
        bbox = draw.textbbox((0, 0), arm_lbl, font=font_row)
        th = bbox[3] - bbox[1]
        ty = row_y + (cell_h - th) // 2
        draw.text((margin + 4, ty), arm_lbl, font=font_row, fill=(20, 20, 20))

        for j, cat_key in enumerate(categories):
            col_x = margin + row_header_left + j * (cell_w + gap)
            img = images[(cat_key, arm_key)]
            scaled_img = img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            # Boost contrast so downsampled monospace strokes stay dark and crisp
            scaled_img = ImageEnhance.Contrast(scaled_img).enhance(2.2)
            canvas.paste(scaled_img, (col_x, row_y))
            draw.rectangle(
                [col_x - 1, row_y - 1, col_x + cell_w, row_y + cell_h],
                outline=(180, 180, 180),
                width=1,
            )

    runs_out = runs_dir / "p1_contact_sheet.png"
    canvas.save(runs_out, dpi=(300, 300))
    report_out = root / "report" / "figures" / "p1_contact_sheet.png"
    canvas.save(report_out, dpi=(300, 300))
    print(f"Saved 12-cell sheet ({total_w}x{total_h}, aspect={total_w/total_h:.2f})")


if __name__ == "__main__":
    build_p1_grid()
