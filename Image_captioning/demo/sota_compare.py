"""Utilities to compare the project model with modern image-captioning baselines.

The module is optional: the core demo still works without Hugging Face models.
Install the extra requirements in requirements.txt to run these comparisons.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image


@dataclass(frozen=True)
class SotaSpec:
    key: str
    display_name: str
    hf_id: str
    family: str
    note: str
    heavy: bool = False


SOTA_MODELS: Dict[str, SotaSpec] = {
    "blip_base": SotaSpec(
        key="blip_base",
        display_name="BLIP-base (Salesforce)",
        hf_id="Salesforce/blip-image-captioning-base",
        family="BLIP",
        note="Captioning model pretrained/fine-tuned on COCO; good practical baseline.",
        heavy=False,
    ),
    "git_base_coco": SotaSpec(
        key="git_base_coco",
        display_name="GIT-base-COCO (Microsoft)",
        hf_id="microsoft/git-base-coco",
        family="GIT",
        note="Generative image-to-text Transformer fine-tuned for COCO captioning.",
        heavy=False,
    ),
    "blip2_opt_2_7b": SotaSpec(
        key="blip2_opt_2_7b",
        display_name="BLIP-2 OPT-2.7B (Salesforce, nặng)",
        hf_id="Salesforce/blip2-opt-2.7b",
        family="BLIP-2",
        note="Very large VLP model with frozen image encoder, Q-Former and OPT language model.",
        heavy=True,
    ),
}


def safe_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def parameter_stats(module_or_modules: Any) -> Dict[str, float]:
    """Return total/trainable params and parameter memory in MB."""
    if isinstance(module_or_modules, (list, tuple)):
        params = []
        for m in module_or_modules:
            params.extend(list(m.parameters()))
    else:
        params = list(module_or_modules.parameters())

    total = sum(p.numel() for p in params)
    trainable = sum(p.numel() for p in params if p.requires_grad)
    frozen = total - trainable
    mem_mb = sum(p.numel() * p.element_size() for p in params) / (1024 ** 2)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(frozen),
        "param_memory_mb": float(mem_mb),
    }


def fmt_int(n: Optional[float]) -> str:
    if n is None:
        return "—"
    try:
        return f"{int(n):,}".replace(",", ".")
    except Exception:
        return str(n)


def fmt_mb(x: Optional[float]) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.1f} MB"
    except Exception:
        return str(x)


def load_sota_model(model_key: str, device: str = "auto", torch_dtype: str = "auto") -> Dict[str, Any]:
    """Load a selected baseline model from Hugging Face Transformers.

    Returns a dict containing processor/model/spec/device.
    """
    if model_key not in SOTA_MODELS:
        raise ValueError(f"Unknown SOTA model key: {model_key}")
    spec = SOTA_MODELS[model_key]
    dev = safe_device(device)

    try:
        from transformers import (
            AutoProcessor,
            AutoModelForCausalLM,
            BlipProcessor,
            BlipForConditionalGeneration,
            Blip2Processor,
            Blip2ForConditionalGeneration,
        )
    except Exception as e:
        raise ImportError(
            "Thiếu thư viện transformers. Chạy: pip install -r requirements.txt"
        ) from e

    dtype = None
    if torch_dtype == "float16" and dev.type == "cuda":
        dtype = torch.float16
    elif torch_dtype == "bfloat16" and dev.type == "cuda":
        dtype = torch.bfloat16

    t0 = time.time()
    if spec.family == "BLIP":
        processor = BlipProcessor.from_pretrained(spec.hf_id)
        kwargs = {"torch_dtype": dtype} if dtype is not None else {}
        model = BlipForConditionalGeneration.from_pretrained(spec.hf_id, **kwargs)
        model = model.to(dev).eval()
    elif spec.family == "GIT":
        processor = AutoProcessor.from_pretrained(spec.hf_id)
        kwargs = {"torch_dtype": dtype} if dtype is not None else {}
        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)
        model = model.to(dev).eval()
    elif spec.family == "BLIP-2":
        processor = Blip2Processor.from_pretrained(spec.hf_id)
        kwargs = {"torch_dtype": dtype} if dtype is not None else {}
        # device_map="auto" is better for BLIP-2, but it complicates memory reporting.
        model = Blip2ForConditionalGeneration.from_pretrained(spec.hf_id, **kwargs)
        model = model.to(dev).eval()
    else:
        raise ValueError(f"Unsupported family: {spec.family}")

    load_ms = (time.time() - t0) * 1000
    stats = parameter_stats(model)
    return {"spec": spec, "processor": processor, "model": model, "device": dev, "load_ms": load_ms, "stats": stats}


def generate_sota_caption(
    loaded_sota: Dict[str, Any],
    image: Image.Image,
    num_beams: int = 5,
    max_new_tokens: int = 40,
) -> Dict[str, Any]:
    spec: SotaSpec = loaded_sota["spec"]
    processor = loaded_sota["processor"]
    model = loaded_sota["model"]
    dev: torch.device = loaded_sota["device"]

    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    with torch.no_grad():
        if spec.family == "BLIP":
            inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(dev)
            generated_ids = model.generate(**inputs, num_beams=num_beams, max_new_tokens=max_new_tokens)
            caption = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        elif spec.family == "GIT":
            inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(dev)
            generated_ids = model.generate(pixel_values=inputs.pixel_values, num_beams=num_beams, max_new_tokens=max_new_tokens)
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        elif spec.family == "BLIP-2":
            inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(dev)
            generated_ids = model.generate(**inputs, num_beams=num_beams, max_new_tokens=max_new_tokens)
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        else:
            raise ValueError(f"Unsupported family: {spec.family}")

    if dev.type == "cuda":
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_mb = None
    infer_ms = (time.time() - t0) * 1000

    return {
        "model_key": spec.key,
        "model": spec.display_name,
        "hf_id": spec.hf_id,
        "caption": caption,
        "infer_ms": infer_ms,
        "peak_vram_mb": peak_mb,
        **loaded_sota["stats"],
    }


def ours_comparison_row(loaded_project: Any, caption: str, infer_ms: float, peak_vram_mb: Optional[float] = None) -> Dict[str, Any]:
    stats = parameter_stats([loaded_project.encoder, loaded_project.decoder])
    return {
        "model_key": "ours",
        "model": "Ours: Dual Encoder + Adaptive + SFS5D + SCST",
        "hf_id": "local checkpoint",
        "caption": caption,
        "infer_ms": infer_ms,
        "peak_vram_mb": peak_vram_mb,
        **stats,
    }


def rows_for_display(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        out.append({
            "Model": r.get("model", ""),
            "Caption": r.get("caption", ""),
            "Params": fmt_int(r.get("total_params")),
            "Trainable": fmt_int(r.get("trainable_params")),
            "Param memory": fmt_mb(r.get("param_memory_mb")),
            "Time/img": f"{r.get('infer_ms', 0):.0f} ms" if r.get("infer_ms") is not None else "—",
            "Peak VRAM": fmt_mb(r.get("peak_vram_mb")),
            "Source": r.get("hf_id", ""),
        })
    return out
