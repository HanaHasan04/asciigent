"""Vision embedding similarity metrics for rendered ASCII art.

Calculates multimodal visual similarity (e.g. CLIP score) between rendered
ASCII bitmaps and target textual concepts, evaluated against scrambled controls
to prevent character-text bias.
"""

from __future__ import annotations

import io
from typing import Sequence

from PIL import Image

from .grid import AsciiGrid
from .render import RenderStyle, render_to_png_bytes

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_DEVICE = None


def get_clip_model(model_name: str = "ViT-B/32"):
    """Lazy load CLIP model on available device (GPU or CPU)."""
    global _CLIP_MODEL, _CLIP_PREPROCESS, _DEVICE
    if _CLIP_MODEL is None:
        import clip
        import torch

        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load(model_name, device=_DEVICE)
        _CLIP_MODEL.eval()
    return _CLIP_MODEL, _CLIP_PREPROCESS, _DEVICE


def _resolve_style(style: RenderStyle | str | None) -> RenderStyle:
    if isinstance(style, RenderStyle):
        return style
    if style == "white_on_black":
        return RenderStyle(fg=(255, 255, 255), bg=(0, 0, 0))
    if style == "black_on_white" or style is None:
        return RenderStyle(fg=(0, 0, 0), bg=(255, 255, 255))
    return RenderStyle()


def compute_clip_score(
    grid: AsciiGrid,
    category: str,
    prompts: Sequence[str] | None = None,
    style: RenderStyle | str | None = None,
) -> float:
    """Compute cosine similarity between rendered grid image and target prompt embeddings."""
    import clip
    import torch

    model, preprocess, device = get_clip_model()
    r_style = _resolve_style(style)

    # 1. Render grid to PNG and load PIL image
    png_bytes = render_to_png_bytes(grid, style=r_style)
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # 2. Text tokens
    if prompts is None:
        prompts = [f"ASCII art of a {category}", f"a drawing of a {category}", f"a {category}"]
    text_tokens = clip.tokenize(list(prompts)).to(device)

    # 3. Compute normalized features
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Average similarity across prompt templates
        similarity = (image_features @ text_features.T).mean().item()

    return float(similarity)


def evaluate_vision_metrics(
    grid: AsciiGrid,
    category: str,
    style: RenderStyle | str | None = None,
) -> dict[str, float]:
    """Compute CLIP visual score for the real drawing and scrambled controls."""
    r_style = _resolve_style(style)
    real_score = compute_clip_score(grid, category, style=r_style)
    scramble_cells_score = compute_clip_score(grid.scrambled("cells", seed=42), category, style=r_style)
    scramble_ink_score = compute_clip_score(grid.scrambled("ink", seed=42), category, style=r_style)

    control_gap = real_score - max(scramble_cells_score, scramble_ink_score)

    return {
        "clip_score_real": round(real_score, 4),
        "clip_score_scramble_cells": round(scramble_cells_score, 4),
        "clip_score_scramble_ink": round(scramble_ink_score, 4),
        "clip_control_gap": round(control_gap, 4),
    }
