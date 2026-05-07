"""Federated learning loop: partitioning → per-client updates → aggregation."""
from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from peft import get_peft_model_state_dict

from .config import ExperimentConfig
from .utils import set_seed, ensure_dir, save_json, get_trainable_params, get_total_params
from .hf_utils import apply_hf_settings
from .data import load_benchmark, build_partition_labels
from .partition import dirichlet_partition
from .models import load_base_model, load_tokenizer
from .lora import apply_late_layer_lora
from .feedback import build_feedback_provider
from .algorithms import get_client_update, FEEDBACK_ALGORITHMS
from .metrics import evaluate_global

def _weighted_avg(states: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
    """FedAvg: weighted average of adapter state-dicts on CPU."""
    pos = [(sd, float(w)) for sd, w in states if float(w) > 0]
    if not pos:
        pos = [(sd, 1.0) for sd, _ in states]

    keys = list(pos[0][0].keys())
    weights = torch.tensor([w for _, w in pos], dtype=torch.float32)
    weights = weights / weights.sum().clamp(min=1e-12)

    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        ts = torch.stack([sd[k].detach().cpu() if sd[k].device.type != "cpu"
                          else sd[k].detach() for sd, _ in pos])
        w = weights.view(-1, *([1] * (ts.dim() - 1)))
        out[k] = (ts * w).sum(dim=0)
    return out

def _server_optimizer_step(
    global_state: Dict[str, torch.Tensor],
    avg_state: Dict[str, torch.Tensor],
    m: Optional[Dict[str, torch.Tensor]],
    v: Optional[Dict[str, torch.Tensor]],
    server_optimizer: str,
    server_lr: float,
    beta1: float,
    beta2: float,
    tau: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    new_global: Dict[str, torch.Tensor] = {}
    new_m: Dict[str, torch.Tensor] = {}
    new_v: Dict[str, torch.Tensor] = {}

    for k in global_state:
        g_k = global_state[k].detach().cpu()              # ← normalize to CPU once
        g = avg_state[k].cpu() - g_k                      # pseudo-gradient
        m_prev = m[k] if (m is not None and k in m) else torch.zeros_like(g)
        m_k = beta1 * m_prev + (1.0 - beta1) * g
        v_prev = v[k] if (v is not None and k in v) else torch.full_like(g, tau ** 2)
        if server_optimizer == "fedadam":
            v_k = beta2 * v_prev + (1.0 - beta2) * g.pow(2)
        elif server_optimizer == "fedyogi":
            v_k = v_prev + (1.0 - beta2) * g.pow(2) * torch.sign(g.pow(2) - v_prev)
        else:
            raise ValueError(f"Unknown server_optimizer: {server_optimizer!r}")

        new_global[k] = g_k + server_lr * m_k / (v_k.sqrt() + tau)  # all CPU
        new_m[k] = m_k
        new_v[k] = v_k

    return new_global, new_m, new_v

def cfg_to_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
    return {name: vars(getattr(cfg, name))
            for name in cfg.__dataclass_fields__
            if hasattr(getattr(cfg, name), "__dataclass_fields__")} | {
        "output_dir": cfg.output_dir,
    }

def run_federated(cfg: ExperimentConfig) -> str:
    set_seed(cfg.data.seed)
    rng = random.Random(cfg.data.seed)
    ensure_dir(cfg.output_dir)
    save_json(os.path.join(cfg.output_dir, "config.json"), cfg_to_dict(cfg))

    hf_info = apply_hf_settings(cfg.hf)
    save_json(os.path.join(cfg.output_dir, "hf_paths.json"), hf_info)

    # Data
    train_ex, eval_ex = load_benchmark(cfg.data, hf=cfg.hf)
    labels = build_partition_labels(train_ex, cfg.data)
    partitions = dirichlet_partition(
        labels, cfg.data.num_clients, cfg.data.dirichlet_alpha,
        cfg.data.seed, min_size=10)

    clients: List[List[Dict[str, Any]]] = []
    for idxs in partitions:
        if cfg.data.max_examples_per_client is not None:
            idxs = idxs[:cfg.data.max_examples_per_client]
        clients.append([train_ex[i] for i in idxs])

    # Model
    tokenizer = load_tokenizer(cfg.model, hf=cfg.hf)
    base = load_base_model(cfg.model, hf=cfg.hf)
    model, lora_info = apply_late_layer_lora(base, cfg.lora)

    # Server optimizer config — read from cfg.fl with safe defaults so that
    # existing configs without these fields continue to work as plain FedAvg.
    server_optimizer: str = getattr(cfg.fl, "aggregation", "fedavg").lower()
    server_lr: float      = float(getattr(cfg.fl, "server_lr",    1e-2))
    server_beta1: float   = float(getattr(cfg.fl, "server_beta1", 0.9))
    server_beta2: float   = float(getattr(cfg.fl, "server_beta2", 0.99))
    server_tau: float     = float(getattr(cfg.fl, "server_tau",   5e-3))

    if server_optimizer not in ("fedavg", "fedadam", "fedyogi"):
        raise ValueError(
            f"cfg.fl.server_optimizer must be 'fedavg', 'fedadam', or 'fedyogi'; "
            f"got {server_optimizer!r}"
        )

    save_json(os.path.join(cfg.output_dir, "model_info.json"), {
        "model_name": cfg.model.model_name,
        "trainable_params": get_trainable_params(model),
        "total_params": get_total_params(model),
        "lora_info": lora_info,
        "num_clients": cfg.data.num_clients,
        "algorithm": cfg.algorithm.name,
        "server_optimizer": server_optimizer,
        **({"server_lr": server_lr, "server_beta1": server_beta1,
            "server_beta2": server_beta2, "server_tau": server_tau}
           if server_optimizer != "fedavg" else {}),
    })

    # Feedback provider (only for algorithms that need it)
    algo = cfg.algorithm.name
    feedback_provider = (build_feedback_provider(cfg.feedback, hf=cfg.hf)
                         if algo in FEEDBACK_ALGORITHMS else None)

    client_update_fn = get_client_update(cfg)
    global_state = get_peft_model_state_dict(model, adapter_name="default")
    torch.save(global_state, os.path.join(cfg.output_dir, "adapter_round_0.pt"))

    # Server-optimizer moment buffers (only used for FedAdam / FedYogi)
    server_m: Optional[Dict[str, torch.Tensor]] = None
    server_v: Optional[Dict[str, torch.Tensor]] = None

    # FL rounds
    history: List[Dict[str, Any]] = []

    for r in tqdm(range(1, cfg.fl.rounds + 1), desc="FL round"):
        t0 = time.time()
        rng_r = random.Random(cfg.data.seed + r)
        selected = rng_r.sample(range(cfg.data.num_clients),
                                k=min(cfg.fl.clients_per_round, cfg.data.num_clients))

        updates: List[Tuple[Dict[str, torch.Tensor], float]] = []
        client_logs: List[Dict[str, Any]] = []

        for cid in tqdm(selected, desc=f"round {r}/{cfg.fl.rounds}", leave=False):
            rng_c = random.Random(cfg.data.seed + r + cid)
            updated, n_eff, logs = client_update_fn(
                model=model, tokenizer=tokenizer,
                global_adapter_state=global_state,
                client_examples=clients[cid],
                cfg=cfg, feedback_provider=feedback_provider,
                rng=rng_c, round=r)
            updates.append((updated, n_eff))
            logs["client_id"] = cid
            logs["n_eff"] = int(n_eff)
            client_logs.append(logs)

        # Aggregation: FedAvg baseline, or server-side adaptive optimizer
        avg_state = _weighted_avg(updates)
        if server_optimizer == "fedavg":
            global_state = avg_state
        else:
            global_state, server_m, server_v = _server_optimizer_step(
                global_state=global_state,
                avg_state=avg_state,
                m=server_m,
                v=server_v,
                server_optimizer=server_optimizer,
                server_lr=server_lr,
                beta1=server_beta1,
                beta2=server_beta2,
                tau=server_tau,
            )

        dt = time.time() - t0

        # Evaluate periodically
        eval_m = None
        do_eval = (eval_ex and (r == 1 or r == cfg.fl.rounds
                                or r % cfg.data.eval_every == 0
                                or r <= cfg.data.eval_every_round_up_to))
        if do_eval:
            torch.save(global_state,
                       os.path.join(cfg.output_dir, f"adapter_round_{r}.pt"))
            eval_m = evaluate_global(model, tokenizer, global_state, eval_ex,
                                     cfg, max_examples=10_000) 

        history.append({"round": r, "selected_clients": selected,
                        "client_logs": client_logs, "eval": eval_m,
                        "seconds": dt})
        save_json(os.path.join(cfg.output_dir, "history.json"),
                  {"history": history})
    return cfg.output_dir