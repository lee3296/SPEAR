from __future__ import annotations

import os
import json
import random
import numpy as np
import torch
# from dataclasses import asdict
from typing import Any, Dict # , Optional

def set_seed(seed: int, deterministic: bool | None = None) -> None:
    """Seed Python/NumPy/PyTorch RNGs and configure determinism.

    Deterministic kernels can be **much** slower for transformer training. For speed,
    leave `deterministic=None` (default) and set `FOCUS_DETERMINISTIC=1` only when you
    explicitly need exact reproducibility.
    For experimental purposes, we run multiple times, so 0 is fine....

    Args:
        seed: RNG seed.
        deterministic: If True, force deterministic algorithms. If None, reads
            `FOCUS_DETERMINISTIC` from the environment (default: False).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic is None:
        det_env = os.environ.get("FOCUS_DETERMINISTIC", "0").strip().lower()
        deterministic = det_env not in ("", "0", "false", "no", "off")

    if deterministic:
        # Required for certain CUDA deterministic paths.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Prefer reproducibility over speed.
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    else:
        # Fast defaults.
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        # TF32 is a major speed win on Ampere+ for matmul-heavy models.
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def dtype_from_str(s: str):
    if s == "bf16":
        return torch.bfloat16
    if s == "fp16":
        return torch.float16
    return torch.float32


def get_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_total_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
