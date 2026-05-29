"""Benchmark project model vs selected image-captioning baselines on a folder of images.

Example:
python benchmark_sota_folder.py \
  --checkpoint checkpoints/checkpoint.pth.tar \
  --word-map checkpoints/word_map.json \
  --yolo checkpoints/yolo12s.pt \
  --image-dir sample_inputs \
  --models blip_base git_base_coco \
  --output benchmark_results.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List

from PIL import Image

from demo_engine import load_model, run_demo_inference
from sota_compare import load_sota_model, generate_sota_caption, ours_comparison_row


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--word-map", required=True)
    p.add_argument("--yolo", default="")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--models", nargs="+", default=["blip_base", "git_base_coco"], help="blip_base git_base_coco blip2_opt_2_7b")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", default="benchmark_results.csv")
    return p.parse_args()


def list_images(folder: str) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [p for p in Path(folder).rglob("*") if p.suffix.lower() in exts]


def main():
    args = parse_args()
    image_paths = list_images(args.image_dir)
    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit("Không tìm thấy ảnh trong --image-dir")

    print("Loading project model...")
    project = load_model(args.checkpoint, args.word_map, device=args.device, yolo_weights=args.yolo or None)

    print("Loading baseline models...")
    baselines = []
    for key in args.models:
        print(f"  - {key}")
        baselines.append(load_sota_model(key, device=args.device))

    rows = []
    for idx, path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] {path.name}")
        img = Image.open(path).convert("RGB")
        res = run_demo_inference(project, img, decode_mode="beam", beam_size=args.beam_size, max_len=args.max_new_tokens)
        ours_caption = res["beam"]["caption"] if res.get("beam") else res["greedy"]["caption"]
        ours = ours_comparison_row(project, ours_caption, res["timing"]["total_ms"])
        ours.update({"image": str(path), "model_key": "ours"})
        rows.append(ours)

        for b in baselines:
            r = generate_sota_caption(b, img, num_beams=args.beam_size, max_new_tokens=args.max_new_tokens)
            r.update({"image": str(path)})
            rows.append(r)

    fieldnames = [
        "image", "model_key", "model", "hf_id", "caption", "total_params", "trainable_params",
        "frozen_params", "param_memory_mb", "infer_ms", "peak_vram_mb"
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
