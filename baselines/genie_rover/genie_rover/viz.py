"""One picture per plan: the camera frame with SAM-TP traversability, and GeNIE's BEV plan beside it."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .planning import Plan


def overlay(rgb: np.ndarray, traversability: np.ndarray) -> np.ndarray:
    """Green where SAM-TP says traversable, red where it does not."""
    image = rgb.astype(np.float32)
    shaded = image.copy()
    shaded[..., 0] = image[..., 0] * 0.55 + 255.0 * (1.0 - traversability) * 0.45
    shaded[..., 1] = image[..., 1] * 0.55 + 255.0 * traversability * 0.45
    shaded[..., 2] = image[..., 2] * 0.55
    return np.clip(shaded, 0, 255).astype(np.uint8)


def composite(rgb: np.ndarray, plan: Plan, caption: str) -> Image.Image:
    left = Image.fromarray(overlay(rgb, plan.traversability))
    height = left.height
    if plan.visualization is not None:
        right = Image.fromarray(np.asarray(plan.visualization, np.uint8)).convert("RGB")
        right = right.resize((max(1, round(right.width * height / right.height)), height))
    else:
        right = Image.new("RGB", (height, height))
    canvas = Image.new("RGB", (left.width + right.width, height + 22), (16, 16, 16))
    canvas.paste(left, (0, 22))
    canvas.paste(right, (left.width, 22))
    ImageDraw.Draw(canvas).text((6, 5), caption, fill=(235, 235, 235))
    return canvas
