from __future__ import annotations

from typing import List, Tuple, Dict
import re

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

from .config import LoraCfg


def _is_linear_like(module: torch.nn.Module) -> bool:
    w = getattr(module, "weight", None)
    if w is None:
        return False
    try:
        return getattr(w, "ndim", None) == 2
    except Exception:
        return False


def _infer_layers_pattern_and_count(model) -> Tuple[str, int]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "model.layers", len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "transformer.h", len(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return "gpt_neox.layers", len(model.gpt_neox.layers)
    if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        return "decoder.layers", len(model.decoder.layers)

    names = [n for n, _ in model.named_modules()]
    prefix_to_idxs: Dict[str, set[int]] = {}

    pat = re.compile(r"^(?P<prefix>.+)\.(?P<idx>\d+)(?:\.|$)")
    for n in names:
        m = pat.match(n)
        if not m:
            continue
        prefix = m.group("prefix").rstrip(".")
        idx = int(m.group("idx"))
        prefix_to_idxs.setdefault(prefix, set()).add(idx)

    candidates = []
    for prefix, idxs in prefix_to_idxs.items():
        if 0 not in idxs:
            continue
        max_idx = max(idxs)
        if any(i not in idxs for i in range(max_idx + 1)):
            continue

        n_layers = max_idx + 1
        bonus = 0
        if prefix.endswith("layers") or prefix.endswith(".h"):
            bonus += 10
        if ".layers" in prefix or ".h" in prefix:
            bonus += 5
        candidates.append((n_layers + bonus, prefix, n_layers))

    if candidates:
        _, best_prefix, best_n = max(candidates, key=lambda x: x[0])
        return best_prefix.rstrip("."), best_n

    return "model.layers", 0


def _auto_target_suffixes(model) -> List[str]:
    """Return suffix names that exist somewhere in the model."""
    suffixes = set()
    for name, module in model.named_modules():
        if _is_linear_like(module):
            suffixes.add(name.split(".")[-1])

    # Prefer groups that match each family
    candidate_groups = [
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # LLaMA/Qwen
        ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],                           # Phi-3
        ["c_attn", "c_proj", "c_fc"],                                                  # GPT-2 style
    ]

    best = []
    best_score = 0
    for g in candidate_groups:
        found = [x for x in g if x in suffixes]
        if len(found) > best_score:
            best = found
            best_score = len(found)

    return best if best else sorted(list(suffixes))[:4]


def _explicit_targets_in_last_layers(
    model,
    layers_pattern: str,
    layers_to_transform: List[int],
    target_suffixes: List[str],
) -> List[str]:
    """
    Build explicit full module names to patch, e.g.:
      model.layers.30.self_attn.q_proj
    """
    suffix_set = set(target_suffixes)
    allowed_prefixes = tuple(f"{layers_pattern}.{i}." for i in layers_to_transform)

    explicit = []
    for name, module in model.named_modules():
        if not name.startswith(allowed_prefixes):
            continue
        if name.split(".")[-1] in suffix_set and _is_linear_like(module):
            explicit.append(name)

    # de-dupe, keep stable order
    seen = set()
    out = []
    for n in explicit:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def apply_late_layer_lora(model, lora_cfg: LoraCfg):
    layers_pattern, num_layers = _infer_layers_pattern_and_count(model)
    layers_pattern = layers_pattern.rstrip(".")

    last_n = int(lora_cfg.last_n_layers)
    if num_layers <= 0:
        layers_to_transform = None
    else:
        start = max(0, num_layers - max(1, last_n))
        layers_to_transform = list(range(start, num_layers))

    # Quantized? prepare (safe to try)
    try:
        use_gc = bool(
            getattr(model, "is_gradient_checkpointing", False)
            or getattr(model, "gradient_checkpointing", False)
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gc)
    except Exception:
        pass

    target_suffixes = lora_cfg.target_modules or _auto_target_suffixes(model)

    if layers_to_transform is None:
        # No layer info -> fall back to suffix-based targeting globally
        target_modules = target_suffixes
    else:
        # IMPORTANT WORKAROUND: build explicit full names and DO NOT use layers_pattern/layers_to_transform
        target_modules = _explicit_targets_in_last_layers(
            model=model,
            layers_pattern=layers_pattern,
            layers_to_transform=layers_to_transform,
            target_suffixes=target_suffixes,
        )

        if not target_modules:
            # As a last resort, patch all linears rather than erroring
            target_modules = "all-linear"

    peft_cfg = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        bias=lora_cfg.bias,
        target_modules=target_modules,   # <-- explicit full module names OR "all-linear"
        task_type=TaskType.CAUSAL_LM,
        # DO NOT pass layers_to_transform/layers_pattern here (workaround)
    )

    model = get_peft_model(model, peft_cfg)

    return model, dict(
        layers_pattern=layers_pattern,
        num_layers=num_layers,
        layers_to_transform=layers_to_transform,
        target_suffixes=target_suffixes,
        target_modules=target_modules,
        targeting_mode="explicit_names" if isinstance(target_modules, list) else "all-linear",
    )