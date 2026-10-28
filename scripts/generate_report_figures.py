"""Generate all figures for the LaTeX report.

Outputs in report/figures/:
1. p1_contact_sheet.png (12 cells: 4 categories x 3 arms)
2. fig2_agentic_turns.png (3-turn progression: Turn 1 Draft -> Turn 2 Whiskers/Paws -> Turn 3 Tail/Collar)
3. fig3_feedback_ablation.png (P2 No-Feedback vs P2 Text-Only vs P2 Visual Perception)
4. fig4_conversational_edit.png (Before vs After centered party hat edit)
5. fig5_animation_strip.png (3 keyframes: blinking eyes + wagging tail, 100% invariant body)
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FIGS = ROOT / "report" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def get_font(size=14, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(name, size)
    except Exception:
        return ImageFont.load_default()


def create_panel_row(
    panels: list[tuple[str, Image.Image]],
    out_path: Path,
    cell_scale: float = 0.75,
    crop_vertical: tuple[float, float] = (0.02, 0.72),
    crop_horizontal: tuple[float, float] = (0.12, 0.82),
):
    """Tiles a list of (caption, image) horizontally with clear headers."""
    n = len(panels)
    scaled = []
    for cap, img in panels:
        img = img.convert("RGB")
        x0 = int(img.width * crop_horizontal[0]) if crop_horizontal else 0
        x1 = int(img.width * crop_horizontal[1]) if crop_horizontal else img.width
        y0 = int(img.height * crop_vertical[0]) if crop_vertical else 0
        y1 = int(img.height * crop_vertical[1]) if crop_vertical else img.height
        img = img.crop((x0, y0, x1, y1))
        w = int(img.width * cell_scale)
        h = int(img.height * cell_scale)
        resized = img.resize((w, h), Image.Resampling.LANCZOS)
        resized = ImageEnhance.Contrast(resized).enhance(2.2)
        scaled.append((cap, resized))

    cell_w = max(im.width for _, im in scaled)
    cell_h = max(im.height for _, im in scaled)
    header_h = 30
    gap = 12
    margin = 10

    total_w = n * cell_w + (n - 1) * gap + 2 * margin
    total_h = cell_h + header_h + 2 * margin

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = get_font(13, bold=True)

    for i, (cap, im) in enumerate(scaled):
        x = margin + i * (cell_w + gap)
        y = margin + header_h
        bbox = draw.textbbox((0, 0), cap, font=font)
        tw = bbox[2] - bbox[0]
        tx = x + (cell_w - tw) // 2
        draw.text((tx, margin + 6), cap, font=font, fill=(25, 25, 25))

        canvas.paste(im, (x, y))
        draw.rectangle([x - 1, y - 1, x + cell_w, y + cell_h], outline=(200, 200, 200), width=1)

    canvas.save(out_path, dpi=(300, 300))
    print(f"Saved: {out_path} ({total_w}x{total_h})")


def make_fig2_agentic():
    p2_dir = RUNS / "20261028T120613-p2-cedeb4"
    t1 = Image.open(p2_dir / "turn_1.png")
    t2 = Image.open(p2_dir / "turn_2.png")
    t3 = Image.open(p2_dir / "turn_3.png")
    panels = [
        ("Turn 1: Initial Draft", t1),
        ("Turn 2: +Whiskers & Paws", t2),
        ("Turn 3: +Tail & Collar", t3),
    ]
    create_panel_row(panels, FIGS / "fig2_agentic_turns.png", cell_scale=0.65, crop_vertical=(0.0, 0.65))


def make_fig3_ablation():
    p2_none = Image.open(RUNS / "20261028T120725-p2-fd6169" / "drawing.png")
    p2_text = Image.open(RUNS / "20261028T120735-p2-c5e6ca" / "drawing.png")
    p2_vis = Image.open(RUNS / "20261028T120613-p2-cedeb4" / "drawing.png")

    panels = [
        ("No-Feedback (Blind)", p2_none),
        ("Text-Only Critique", p2_text),
        ("Visual Perception (Ours)", p2_vis),
    ]
    create_panel_row(panels, FIGS / "fig3_feedback_ablation.png", cell_scale=0.65, crop_vertical=(0.0, 0.82))


def make_fig4_edit():
    p4_dir = RUNS / "p4_edit"
    before = Image.open(p4_dir / "before.png")
    after = Image.open(p4_dir / "after.png")
    panels = [
        ("Before: Base Cat", before),
        ("After: +Party Hat (Rows 1-4)", after),
    ]
    create_panel_row(panels, FIGS / "fig4_conversational_edit.png", cell_scale=0.70, crop_vertical=(0.0, 0.58))


def make_fig5_animate():
    p6_dir = RUNS / "p6_animate"
    f0 = Image.open(p6_dir / "frame_000.png")
    f1 = Image.open(p6_dir / "frame_001.png")
    f2 = Image.open(p6_dir / "frame_002.png")
    panels = [
        ("Frame 1 (o.o, Tail Up)", f0),
        ("Frame 2 (-.-, Tail Mid)", f1),
        ("Frame 3 (^.^, Tail Right)", f2),
    ]
    create_panel_row(panels, FIGS / "fig5_animation_strip.png", cell_scale=0.65, crop_vertical=(0.0, 0.65))


if __name__ == "__main__":
    make_fig2_agentic()
    make_fig3_ablation()
    make_fig4_edit()
    make_fig5_animate()
    print("All report figures assembled successfully!")
