"""Inference + tracing utilities for the Image Captioning demo.

This module is designed to work with checkpoints saved by the user's project:
- checkpoint['encoder'] is DualStreamEncoder
- checkpoint['decoder'] is DecoderAdaptive
- WORDMAP_*.json contains token -> id

The important addition here is tracing: we keep the attention vector at every
word-generation step so the Streamlit app can visualize *why* a word was chosen.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.ops as ops
from PIL import Image

from models import TOP_K_OBJECTS


@dataclass
class LoadedModel:
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    word_map: Dict[str, int]
    rev_word_map: Dict[int, str]
    device: torch.device
    checkpoint_meta: Dict[str, Any]


def _safe_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def load_model(
    checkpoint_path: str | Path,
    word_map_path: str | Path,
    device: str = "auto",
    yolo_weights: Optional[str] = None,
) -> LoadedModel:
    """Load encoder/decoder and word map.

    Important: checkpoints in this project store entire module objects, so the
    local models.py must be importable before torch.load().
    """
    checkpoint_path = Path(checkpoint_path)
    word_map_path = Path(word_map_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")
    if not word_map_path.exists():
        raise FileNotFoundError(f"Không tìm thấy word map: {word_map_path}")

    if yolo_weights:
        os.environ["YOLO_WEIGHTS"] = str(yolo_weights)

    dev = _safe_device(device)
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    encoder = ckpt["encoder"].to(dev).eval()
    decoder = ckpt["decoder"].to(dev).eval()

    with open(word_map_path, "r", encoding="utf-8") as f:
        word_map = json.load(f)
    rev_word_map = {int(v): k for k, v in word_map.items()}

    meta = {
        "epoch": ckpt.get("epoch", None),
        "bleu4": ckpt.get("bleu-4", None),
        "meteor": ckpt.get("meteor", None),
        "rouge_l": ckpt.get("rouge-l", None),
        "device": str(dev),
        "vocab_size": len(word_map),
        "encoder_type": encoder.__class__.__name__,
        "decoder_type": decoder.__class__.__name__,
        "encoded_image_size": getattr(encoder, "enc_image_size", None),
        "top_k_objects": TOP_K_OBJECTS,
    }
    return LoadedModel(encoder, decoder, word_map, rev_word_map, dev, meta)


def preprocess_image(pil_img: Image.Image, image_size: int = 256) -> Tuple[torch.Tensor, Image.Image]:
    """Return normalized tensor [1,3,H,W] and the resized RGB image shown in demo."""
    rgb = pil_img.convert("RGB")
    resized = rgb.resize((image_size, image_size), Image.Resampling.BILINEAR)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(resized).unsqueeze(0)
    return tensor, resized


def _postprocessed_yolo_boxes(encoder, yolo_images: torch.Tensor, conf: float = 0.25) -> Tuple[Optional[List[torch.Tensor]], Optional[List[List[str]]], Optional[List[List[float]]]]:
    """Get boxes through Ultralytics post-processing instead of manually parsing raw tensors.

    The raw tensor layout can change across YOLO/Ultralytics versions. Reading
    raw_output as [cx, cy, w, h, conf, ...] may create fake boxes near (0, 0).
    For demo/visualization, use the official post-processed Results object.
    """
    wrapper_list = getattr(encoder, "_yolo_wrapper", None)
    yolo = wrapper_list[0] if wrapper_list else None
    if yolo is None:
        return None, None, None

    device = yolo_images.device
    H, W = yolo_images.shape[-2], yolo_images.shape[-1]
    try:
        # This performs a second YOLO pass, but boxes are decoded + NMS-filtered
        # by Ultralytics, which is much safer for a public demo.
        results = yolo.predict(
            yolo_images.detach(),
            imgsz=max(H, W),
            conf=conf,
            iou=0.70,
            verbose=False,
            device=str(device) if device.type == "cpu" else 0,
        )
        names = getattr(yolo, "names", {}) or getattr(getattr(yolo, "model", None), "names", {}) or {}

        boxes_per_img: List[torch.Tensor] = []
        labels_per_img: List[List[str]] = []
        confs_per_img: List[List[float]] = []
        for r in results:
            if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
                boxes_per_img.append(torch.zeros(0, 4, device=device))
                labels_per_img.append([])
                confs_per_img.append([])
                continue

            b = r.boxes.xyxy.detach().to(device).float()
            c = r.boxes.conf.detach().to(device).float() if getattr(r.boxes, "conf", None) is not None else torch.ones(len(b), device=device)
            cls = r.boxes.cls.detach().to(device).long() if getattr(r.boxes, "cls", None) is not None else torch.full((len(b),), -1, device=device)

            # Safety: some backends may return normalized boxes. Convert if needed.
            if len(b) and float(b.max().item()) <= 1.5:
                b[:, [0, 2]] *= float(W)
                b[:, [1, 3]] *= float(H)
            b[:, [0, 2]] = b[:, [0, 2]].clamp(0, W)
            b[:, [1, 3]] = b[:, [1, 3]].clamp(0, H)

            order = torch.argsort(c, descending=True)
            b = b[order]
            c = c[order]
            cls = cls[order]

            labels = []
            for k in cls.detach().cpu().tolist():
                labels.append(str(names.get(int(k), f"class_{int(k)}")) if int(k) >= 0 else "object")

            boxes_per_img.append(b)
            labels_per_img.append(labels)
            confs_per_img.append([float(x) for x in c.detach().cpu().tolist()])
        return boxes_per_img, labels_per_img, confs_per_img
    except Exception as e:
        print(f"[WARN] Ultralytics postprocess boxes failed, fallback to raw parser: {e}")
        return None, None, None


def _extract_yolo_feature_map_and_boxes(encoder, images: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[List[str]], List[List[float]], str]:
    """Run YOLO branch and return feature map + reliable boxes.

    The feature map is captured from the YOLO core for ROI Align. Boxes are
    obtained with Ultralytics' official post-processing whenever possible; this
    avoids fake top-left boxes caused by raw-output format mismatch.
    """
    device = images.device
    batch_size = images.size(0)
    H, W = images.shape[-2], images.shape[-1]

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    yolo_images = (images * std + mean).clamp(0.0, 1.0)

    hook_feats: List[torch.Tensor] = []
    handle = encoder.yolo_core.model[-2].register_forward_hook(lambda m, inp, o: hook_feats.append(o))
    with torch.no_grad():
        raw_output = encoder.yolo_core(yolo_images)
    handle.remove()

    f_map = hook_feats[0]
    if isinstance(f_map, (list, tuple)):
        f_map = f_map[1] if len(f_map) > 1 else f_map[0]
    f_map = f_map.detach().float()

    boxes_per_img, labels_per_img, confs_per_img = _postprocessed_yolo_boxes(encoder, yolo_images, conf=0.25)
    box_source = "ultralytics_postprocess"
    if boxes_per_img is None:
        boxes_per_img = encoder._parse_yolo12_boxes_batched(raw_output, batch_size, (H, W), device)
        labels_per_img = [["object"] * (0 if b is None else len(b)) for b in boxes_per_img]
        confs_per_img = [[0.0] * (0 if b is None else len(b)) for b in boxes_per_img]
        box_source = "raw_tensor_fallback"

    return f_map, boxes_per_img, raw_output, labels_per_img or [], confs_per_img or [], box_source


def encode_with_trace(loaded: LoadedModel, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Run encoder while recording global slots, object boxes, and masks."""
    encoder = loaded.encoder
    device = loaded.device
    images = image_tensor.to(device)
    batch_size = images.size(0)
    H, W = images.shape[-2], images.shape[-1]

    with torch.no_grad():
        # Global stream: ResNet -> adaptive pool -> flatten slots
        gf_map = encoder.resnet(images)
        gf_pooled = encoder.adaptive_pool(gf_map)
        grid_h, grid_w = int(gf_pooled.shape[-2]), int(gf_pooled.shape[-1])
        gf = gf_pooled.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2048)
        gf = gf + encoder.global_type_emb

        # Local/object stream: YOLO boxes -> ROI Align -> projection -> spatial fusion
        f_map, boxes_per_img, _, labels_per_img, confs_per_img, box_source = _extract_yolo_feature_map_and_boxes(encoder, images)
        spatial_scale = float(f_map.shape[-1]) / W

        roi_list, real_boxes_list, n_per_img = [], [], []
        for boxes in boxes_per_img:
            b = boxes[:TOP_K_OBJECTS] if (boxes is not None and len(boxes) > 0) else torch.zeros(0, 4, device=device)
            roi_list.append(b)
            real_boxes_list.append(b)
            n_per_img.append(int(len(b)))

        obj_vecs = torch.zeros(batch_size, TOP_K_OBJECTS, 2048, device=device)
        is_padding = torch.ones(batch_size, TOP_K_OBJECTS, dtype=torch.bool, device=device)

        if any(n > 0 for n in n_per_img):
            roi_out = ops.roi_align(f_map, roi_list, output_size=(1, 1), spatial_scale=spatial_scale)
            idx = 0
            for i, n in enumerate(n_per_img):
                if n > 0:
                    vecs = roi_out[idx: idx + n].view(n, -1)
                    vecs = encoder.yolo_proj(vecs)
                    vecs = encoder.spatial_fusion(vecs, real_boxes_list[i], H, W)
                    obj_vecs[i, :n] = vecs
                    is_padding[i, :n] = False
                idx += n

        real = ~is_padding
        if real.any():
            obj_vecs[real] = obj_vecs[real] + encoder.local_type_emb.view(2048)

        combined = torch.cat([gf, obj_vecs], dim=1)
        global_pad = torch.zeros(batch_size, gf.size(1), dtype=torch.bool, device=device)
        full_pad_mask = torch.cat([global_pad, is_padding], dim=1)

    trace = {
        "input_shape": tuple(images.shape),
        "global_feature_map_shape": tuple(gf_map.shape),
        "global_grid": (grid_h, grid_w),
        "global_slots": int(gf.size(1)),
        "object_slots": TOP_K_OBJECTS,
        "valid_objects": n_per_img,
        "boxes": [b.detach().cpu().numpy().tolist() for b in real_boxes_list],
        "box_labels": [labels[:TOP_K_OBJECTS] for labels in labels_per_img],
        "box_confidences": [confs[:TOP_K_OBJECTS] for confs in confs_per_img],
        "box_source": box_source,
        "padding_mask": full_pad_mask.detach().cpu().numpy().astype(bool).tolist(),
        "encoder_out_shape": tuple(combined.shape),
        "yolo_feature_map_shape": tuple(f_map.shape),
    }
    return combined, full_pad_mask, trace


def _decode_step(decoder, encoder_out, h, c, prev_word_emb, padding_mask):
    g_t = torch.sigmoid(decoder.sentinel_w_x(prev_word_emb) + decoder.sentinel_w_h(h))
    s_t = g_t * torch.tanh(c)
    context, alpha = decoder.attention(encoder_out, h, s_t, padding_mask)
    h, c = decoder.decode_step(torch.cat([prev_word_emb, context], dim=1), (h, c))
    logits = decoder.fc(h)
    log_probs = F.log_softmax(logits, dim=1)
    probs = torch.softmax(logits, dim=1)
    return logits, probs, log_probs, h, c, alpha, context, g_t, s_t


def greedy_decode_with_trace(
    loaded: LoadedModel,
    encoder_out: torch.Tensor,
    padding_mask: torch.Tensor,
    max_len: int = 50,
    topk_words: int = 5,
) -> Dict[str, Any]:
    """Greedy decoding with per-token attention trace."""
    decoder = loaded.decoder
    word_map = loaded.word_map
    rev = loaded.rev_word_map
    device = loaded.device

    start_token = word_map["<start>"]
    end_token = word_map["<end>"]
    skip = {start_token, end_token, word_map.get("<pad>", 0)}

    tokens: List[int] = []
    steps: List[Dict[str, Any]] = []

    with torch.no_grad():
        h, c = decoder.init_hidden_state(encoder_out)
        prev_word = torch.tensor([start_token], dtype=torch.long, device=device)

        for t in range(max_len):
            emb = decoder.embedding(prev_word)
            logits, probs, log_probs, h, c, alpha, context, g_t, s_t = _decode_step(
                decoder, encoder_out, h, c, emb, padding_mask
            )
            token = int(log_probs.argmax(dim=1).item())
            prob = float(probs[0, token].item())
            topv, topi = probs[0].topk(min(topk_words, probs.shape[1]))
            top_alternatives = [
                {"word": rev.get(int(idx), "<unk>"), "prob": float(val)}
                for val, idx in zip(topv.detach().cpu().tolist(), topi.detach().cpu().tolist())
            ]

            alpha_np = alpha[0].detach().float().cpu().numpy()
            steps.append({
                "t": t + 1,
                "token_id": token,
                "word": rev.get(token, "<unk>"),
                "prob": prob,
                "log_prob": float(log_probs[0, token].item()),
                "top_alternatives": top_alternatives,
                "alpha": alpha_np.tolist(),
                "sentinel_alpha": float(alpha_np[-1]),
                "context_norm": float(context.norm(dim=1).item()),
                "gate_mean": float(g_t.mean().item()),
            })

            if token == end_token:
                break
            if token not in skip:
                tokens.append(token)
            prev_word = torch.tensor([token], dtype=torch.long, device=device)

    words = [rev.get(t, "<unk>") for t in tokens]
    caption = " ".join(words)
    avg_prob = float(np.mean([s["prob"] for s in steps if s["word"] != "<end>"])) if steps else 0.0
    return {"caption": caption, "tokens": tokens, "words": words, "steps": steps, "avg_token_prob": avg_prob}


def _length_penalty(length: int, alpha: float) -> float:
    if alpha == 0.0:
        return 1.0
    return ((5.0 + length) / 6.0) ** alpha


def beam_decode(
    loaded: LoadedModel,
    encoder_out: torch.Tensor,
    padding_mask: torch.Tensor,
    beam_size: int = 5,
    max_len: int = 50,
    length_penalty_alpha: float = 0.7,
) -> Dict[str, Any]:
    """Beam search final caption. Heatmap is shown for greedy path for clarity."""
    decoder = loaded.decoder
    word_map = loaded.word_map
    rev = loaded.rev_word_map
    device = loaded.device
    k = int(beam_size)
    start_token = word_map["<start>"]
    end_token = word_map["<end>"]
    pad_token = word_map.get("<pad>", 0)
    skip = {start_token, end_token, pad_token}
    vocab_size = len(word_map)

    with torch.no_grad():
        N = encoder_out.size(1)
        encoder_out_k = encoder_out.expand(k, N, encoder_out.size(2)).contiguous()
        padding_mask_k = padding_mask.expand(k, N).contiguous()
        h, c = decoder.init_hidden_state(encoder_out_k)

        seqs = torch.full((k, 1), start_token, dtype=torch.long, device=device)
        top_scores = torch.zeros(k, dtype=torch.float, device=device)
        complete_seqs: List[List[int]] = []
        complete_scores: List[float] = []
        complete_lens: List[int] = []
        s = k

        for step in range(max_len):
            prev_word = seqs[:, -1]
            emb = decoder.embedding(prev_word)
            _, _, log_probs, h, c, _, _, _, _ = _decode_step(
                decoder, encoder_out_k[:s], h[:s], c[:s], emb, padding_mask_k[:s]
            )
            scores = top_scores[:s].unsqueeze(1) + log_probs
            if step == 0:
                top_scores_new, top_words = scores[0].topk(k, dim=0)
            else:
                top_scores_new, top_words = scores.view(-1).topk(k, dim=0)

            beam_idx = torch.div(top_words, vocab_size, rounding_mode="floor")
            token_idx = top_words % vocab_size
            seqs_new = torch.cat([seqs[beam_idx], token_idx.unsqueeze(1)], dim=1)

            complete_mask = token_idx == end_token
            incomplete_mask = ~complete_mask
            for j in range(k):
                if complete_mask[j]:
                    seq_tokens = [int(t) for t in seqs_new[j, 1:].tolist() if int(t) not in skip]
                    complete_seqs.append(seq_tokens)
                    complete_scores.append(float(top_scores_new[j].item()))
                    complete_lens.append(len(seq_tokens))

            inc_idx = incomplete_mask.nonzero(as_tuple=False).squeeze(1)
            if len(inc_idx) == 0:
                break
            s = len(inc_idx)
            seqs = seqs_new[inc_idx]
            top_scores = top_scores_new[inc_idx]
            h = h[beam_idx[inc_idx]]
            c = c[beam_idx[inc_idx]]
            encoder_out_k = encoder_out_k[beam_idx[inc_idx]]
            padding_mask_k = padding_mask_k[beam_idx[inc_idx]]

        if complete_seqs:
            normalized = [score / _length_penalty(length, length_penalty_alpha) for score, length in zip(complete_scores, complete_lens)]
            best = int(np.argmax(normalized))
            seq = complete_seqs[best]
            score = normalized[best]
        else:
            seq = [int(t) for t in seqs[0, 1:].tolist() if int(t) not in skip]
            score = float(top_scores[0].item()) if len(top_scores) else 0.0

    words = [rev.get(t, "<unk>") for t in seq]
    return {"caption": " ".join(words), "tokens": seq, "words": words, "score": float(score)}


def run_demo_inference(
    loaded: LoadedModel,
    pil_img: Image.Image,
    decode_mode: str = "both",
    beam_size: int = 5,
    max_len: int = 50,
    length_penalty_alpha: float = 0.7,
) -> Dict[str, Any]:
    """End-to-end inference from PIL image."""
    image_tensor, resized = preprocess_image(pil_img)
    t0 = time.time()
    encoder_out, padding_mask, enc_trace = encode_with_trace(loaded, image_tensor)
    t1 = time.time()

    greedy = greedy_decode_with_trace(loaded, encoder_out, padding_mask, max_len=max_len)
    t2 = time.time()
    beam = None
    if decode_mode in {"both", "beam"}:
        beam = beam_decode(loaded, encoder_out, padding_mask, beam_size=beam_size, max_len=max_len, length_penalty_alpha=length_penalty_alpha)
    t3 = time.time()

    return {
        "image": resized,
        "encoder_out": encoder_out.detach().cpu(),
        "padding_mask": padding_mask.detach().cpu(),
        "encoder_trace": enc_trace,
        "greedy": greedy,
        "beam": beam,
        "timing": {
            "encode_ms": (t1 - t0) * 1000,
            "greedy_ms": (t2 - t1) * 1000,
            "beam_ms": (t3 - t2) * 1000 if beam is not None else 0.0,
            "total_ms": (t3 - t0) * 1000,
        },
    }
