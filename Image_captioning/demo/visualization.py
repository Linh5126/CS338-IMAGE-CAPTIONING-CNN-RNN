"""Visualization helpers for attention heatmaps and YOLO boxes."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)


def global_attention_map(alpha: List[float], global_grid: Tuple[int, int], image_size: Tuple[int, int]) -> np.ndarray:
    gh, gw = global_grid
    n = gh * gw
    arr = np.array(alpha[:n], dtype=np.float32).reshape(gh, gw)
    arr = _normalize(arr)
    im = Image.fromarray((arr * 255).astype(np.uint8)).resize(image_size, Image.Resampling.BILINEAR)
    return np.asarray(im).astype(np.float32) / 255.0


def object_attention_map(
    alpha: List[float],
    boxes: List[List[float]],
    global_slots: int,
    image_size: Tuple[int, int],
    model_size: int = 256,
) -> np.ndarray:
    W, H = image_size
    heat = np.zeros((H, W), dtype=np.float32)
    if not boxes:
        return heat
    object_alpha = np.array(alpha[global_slots: global_slots + len(boxes)], dtype=np.float32)
    for a, b in zip(object_alpha, boxes):
        x1, y1, x2, y2 = b
        x1 = int(max(0, min(W - 1, x1 / model_size * W)))
        x2 = int(max(0, min(W, x2 / model_size * W)))
        y1 = int(max(0, min(H - 1, y1 / model_size * H)))
        y2 = int(max(0, min(H, y2 / model_size * H)))
        if x2 > x1 and y2 > y1:
            heat[y1:y2, x1:x2] += float(a)
    return _normalize(heat)


def combined_attention_map(alpha, trace, image_size: Tuple[int, int], global_weight: float = 0.65) -> Dict[str, np.ndarray]:
    gmap = global_attention_map(alpha, tuple(trace["global_grid"]), image_size)
    boxes = trace["boxes"][0] if trace.get("boxes") else []
    omap = object_attention_map(alpha, boxes, int(trace["global_slots"]), image_size)
    combo = _normalize(global_weight * gmap + (1.0 - global_weight) * omap)
    return {"global": gmap, "object": omap, "combined": combo}


def overlay_heatmap(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay a simple red-yellow heatmap without requiring OpenCV."""
    base = pil_img.convert("RGB")
    W, H = base.size
    hm = Image.fromarray((_normalize(heatmap) * 255).astype(np.uint8)).resize((W, H), Image.Resampling.BILINEAR)
    hm_arr = np.asarray(hm).astype(np.float32) / 255.0

    # Custom thermal map: dark transparent -> yellow -> red.
    red = np.clip(2.0 * hm_arr, 0, 1)
    green = np.clip(2.0 * hm_arr - 0.35, 0, 1)
    blue = np.clip(0.25 * hm_arr, 0, 0.25)
    color = np.stack([red, green, blue], axis=-1) * 255.0

    base_arr = np.asarray(base).astype(np.float32)
    out = (1 - alpha * hm_arr[..., None]) * base_arr + (alpha * hm_arr[..., None]) * color
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def draw_boxes(pil_img: Image.Image, boxes: List[List[float]], scores: List[float] | None = None, model_size: int = 256) -> Image.Image:
    im = pil_img.convert("RGB").copy()
    W, H = im.size
    draw = ImageDraw.Draw(im)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b
        x1 = x1 / model_size * W
        x2 = x2 / model_size * W
        y1 = y1 / model_size * H
        y2 = y2 / model_size * H
        width = 3
        draw.rectangle([x1, y1, x2, y2], outline=(30, 136, 229), width=width)
        label = f"obj {i+1}"
        if scores is not None and i < len(scores):
            label += f" · {scores[i]:.3f}"
        tx, ty = x1 + 3, max(0, y1 - 18)
        draw.rectangle([tx - 2, ty - 2, tx + 95, ty + 16], fill=(30, 136, 229))
        draw.text((tx, ty), label, fill=(255, 255, 255))
    return im


def attention_breakdown(alpha: List[float], trace: Dict) -> Dict[str, float]:
    a = np.array(alpha, dtype=np.float32)
    g = int(trace["global_slots"])
    o = int(trace["object_slots"])
    global_sum = float(a[:g].sum())
    object_sum = float(a[g:g+o].sum())
    sentinel = float(a[-1])
    return {"global": global_sum, "object": object_sum, "sentinel": sentinel}
