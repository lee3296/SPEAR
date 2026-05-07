"""Algorithm dispatch.

This project supports 5 algorithms:
  - SPEAR:    our method (implemented in spear.py)

Baselines:
  - GRPO:     Group Relative Policy Optimisation (multi-sample RL)
  - FEEDBACK_SFT: SFT with feedback-augmented prompts (no distillation, just MLE on y0)
  - OPSD:     On-Policy Self-Distillation (privileged teacher context)
  - RLTF-SD:  RL from Text Feedback via self-distillation

All client updates follow the same FL interface:
  (model, tokenizer, global_adapter_state, client_examples, cfg, feedback_provider, rng, round)
  -> (updated_adapter_state, n_eff, logs)
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from .config import ExperimentConfig
from .spear import client_spear_update, is_correct, build_revision_prompt
from .inference import GenParams, generate_batch_hf
from .prompting import format_prompt
from .spear import _clear_gpu, _chunked, strict_win_ok, build_format_fix_prompt
from .spear import WinTrace, _build_sft_batch


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _cosine_lr(cur, total, peak, min_lr, warmup_ratio, use_warmup):
    if not use_warmup:
        return peak
    w = int(warmup_ratio * total)
    if cur <= w:
        return peak * cur / max(1, w)
    p = (cur - w) / max(1, total - w)
    return min_lr + (peak - min_lr) * 0.5 * (1 + math.cos(math.pi * p))


def _sample(rng, items, k):
    if not items or k <= 0:
        return []
    return rng.sample(items, k) if len(items) >= k else [rng.choice(items) for _ in range(k)]


def _truncate_text(text: Any, max_chars: int) -> str:
    s = "" if text is None else str(text)
    if max_chars is None or max_chars <= 0:
        return s
    return s[:max_chars]

def _build_completion_batch(tokenizer, prompts_raw, completions, max_seq_len):
    """Build batch for distillation / GRPO with completion mask.

    Returns (input_ids, attn_mask, completion_mask) where completion_mask
    aligns with logits[:, :-1] and is 1.0 on completion tokens only.
    """
    B = len(prompts_raw)
    prompts = [format_prompt(tokenizer, p) for p in prompts_raw]
    p_enc = tokenizer(prompts, add_special_tokens=False, truncation=False)
    c_enc = tokenizer(completions, add_special_tokens=False, truncation=False)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id or 0

    seqs, masks, Lmax = [], [], 0
    for p_ids, c_ids in zip(p_enc["input_ids"], c_enc["input_ids"]):
        c = list(c_ids) + ([eos_id] if eos_id is not None else [])
        c = c[:max_seq_len - 1]
        p = list(p_ids)[-(max_seq_len - len(c)) :]
        seq = p + c
        if len(seq) < 2:
            seq = (seq + [eos_id or 0, eos_id or 0])[:2]
        L = len(seq)
        m = [0.0] * (L - 1)
        for t in range(max(0, len(p) - 1), L - 1):
            m[t] = 1.0
        seqs.append(seq)
        masks.append(m)
        Lmax = max(Lmax, L)

    Tmax = max(1, Lmax - 1)
    ids = torch.full((B, Lmax), pad_id, dtype=torch.long)
    attn = torch.zeros((B, Lmax), dtype=torch.long)
    mask_out = torch.zeros((B, Tmax), dtype=torch.float32)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        L = len(s)
        ids[i, -L:] = torch.tensor(s)
        attn[i, -L:] = 1
        mask_out[i, -len(m):] = torch.tensor(m)
    return ids, attn, mask_out


def _pad_left(x: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
    if int(x.shape[1]) >= int(target_len):
        return x
    pad = x.new_full((x.shape[0], int(target_len) - int(x.shape[1])), pad_value)
    return torch.cat([pad, x], dim=1)


def _pad_left_mask(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if int(x.shape[1]) >= int(target_len):
        return x
    pad = x.new_zeros((x.shape[0], int(target_len) - int(x.shape[1])))
    return torch.cat([pad, x], dim=1)


def _align_student_teacher_batches(tokenizer, s_ids, s_attn, s_mask, t_ids, t_attn, t_mask):
    """Left-pad student/teacher batches to a common sequence length.

    Completion masks are aligned at the end (because we use left padding).
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    L = max(int(s_ids.shape[1]), int(t_ids.shape[1]))
    s_ids = _pad_left(s_ids, L, pad_id)
    t_ids = _pad_left(t_ids, L, pad_id)
    s_attn = _pad_left(s_attn, L, 0)
    t_attn = _pad_left(t_attn, L, 0)
    T = max(int(s_mask.shape[1]), int(t_mask.shape[1]))
    s_mask = _pad_left_mask(s_mask, T)
    t_mask = _pad_left_mask(t_mask, T)
    # Ensure the mask length matches logits length (L-1)
    target_T = max(1, L - 1)
    s_mask = s_mask[:, -target_T:]
    t_mask = t_mask[:, -target_T:]
    return s_ids, s_attn, s_mask, t_ids, t_attn, t_mask


def _distill_kl_loss(student_logits, teacher_logits, mask, T: float) -> torch.Tensor:
    """Token-masked KL(teacher || student) over completion positions."""
    # logits are (B, seq-1, V)
    qT = F.softmax(teacher_logits / T, dim=-1)
    log_qS = F.log_softmax(student_logits / T, dim=-1)
    kl = F.kl_div(log_qS, qT, reduction="none").sum(-1)
    m = mask.to(kl.dtype)
    return (kl * m).sum() / m.sum().clamp(min=1.0) * (T * T)


# ---------------------------------------------------------------------------
# Baseline prompt builders
# ---------------------------------------------------------------------------

def _privileged_text_from_example(ex: Dict[str, Any], max_chars: int) -> str:
    meta = ex.get("meta", {})
    priv = (
        meta.get("answer")
        or meta.get("solution")
        or meta.get("complex_cot")
        or meta.get("reference")
        or meta.get("targets")
        or meta.get("final_answer")
        or ""
    )
    # For math_mcqa: append the step-by-step solution additively. The `or` chain
    # above stops at meta["answer"] (the letter), so solution would otherwise
    # never be included.
    solution = str(meta.get("solution") or "")
    if solution and solution != priv:
        priv = (f"{priv}\n\nSolution:\n{solution}" if priv
                else f"Solution:\n{solution}")
    # For StrategyQA: append supporting facts additively.
    facts = list(meta.get("facts") or [])
    if facts:
        facts_text = "\n".join(f"  - {f}" for f in facts)
        priv = (f"{priv}\n\nSupporting facts:\n{facts_text}" if priv
                else f"Supporting facts:\n{facts_text}")
    return _truncate_text(priv, max_chars)


def _opsd_teacher_prompt(prompt: str, priv: str) -> str:
    return (
        "[PRIVILEGED INFORMATION]\n"
        f"{priv}\n\n"
        "[TASK]\n"
        f"{prompt}\n"
    )


# ---------------------------------------------------------------------------
# OPSD
# ---------------------------------------------------------------------------

def client_opsd_update(model, tokenizer, global_adapter_state, client_examples,
                       cfg, feedback_provider, rng, round):
    """On-Policy Self-Distillation baseline (OPSD).

    Student generates y0 from the task prompt. Teacher is the *same* model but
    conditioned on privileged text (gold trace/solution/label) and provides dense
    token-level supervision via KL distillation over the student's rollout.
    """
    set_peft_model_state_dict(model, global_adapter_state, adapter_name="default")
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n and ".default." in n)

    # Rollout
    n_rollout = min(len(client_examples), max(1, cfg.fl.local_rollout_steps))
    rollout_ex = (rng.sample(client_examples, n_rollout)
                  if len(client_examples) >= n_rollout else client_examples)
    prompts = [ex["prompt"] for ex in rollout_ex]
    gen_p = GenParams(max_new_tokens=cfg.gen.max_new_tokens_y0,
                      do_sample=cfg.gen.do_sample,
                      temperature=cfg.gen.temperature,
                      top_p=cfg.gen.top_p,
                      use_cache=True)
    y0: List[str] = []
    bsz_roll = max(1, cfg.gen.rollout_batch_size)
    model.eval(); model.set_adapter("default")
    for i in range(0, len(prompts), bsz_roll):
        y0.extend(generate_batch_hf(model, tokenizer, prompts[i:i+bsz_roll], gen_p))

    priv_max = int(getattr(cfg.distill, "priv_max_chars", 2000))
    teacher_prompts = [_opsd_teacher_prompt(ex["prompt"], _privileged_text_from_example(ex, priv_max))
                       for ex in rollout_ex]
    traces = list(zip(prompts, y0, teacher_prompts))
    if not traces:
        return global_adapter_state, len(client_examples), {"algo": "opsd", "steps": 0}

    # Optim
    lr = _cosine_lr(round, cfg.fl.rounds, cfg.optim.lr, cfg.optim.lr_min,
                    cfg.optim.warmup_ratio, cfg.optim.use_warmup)
    opt = AdamW([p for p in model.parameters() if p.requires_grad],
                lr=lr, weight_decay=cfg.optim.weight_decay)

    device = model.device
    amp = cfg.model.dtype in ("bf16", "fp16") and torch.cuda.is_available()
    amp_dt = torch.bfloat16 if cfg.model.dtype == "bf16" else torch.float16
    grad_accum = max(1, cfg.optim.grad_accum_steps)
    bsz = max(1, cfg.optim.batch_size)
    micro_steps = max(1, cfg.fl.local_train_steps) * grad_accum
    T = float(getattr(cfg.distill, "temperature", 2.0))

    prev_cache = getattr(model.config, "use_cache", None)
    if prev_cache is not None:
        model.config.use_cache = False

    logs = {"algo": "opsd", "loss": 0.0, "kl": 0.0, "steps": 0} 

    try:
        model.train(); model.set_adapter("default")
        for step in range(micro_steps):
            batch = _sample(rng, traces, bsz)
            s_prompts = [b[0] for b in batch]
            comps = [b[1] for b in batch]
            t_prompts = [b[2] for b in batch]

            s_ids, s_attn, s_mask = _build_completion_batch(tokenizer, s_prompts, comps, cfg.model.max_seq_len)
            t_ids, t_attn, t_mask = _build_completion_batch(tokenizer, t_prompts, comps, cfg.model.max_seq_len)
            s_ids, s_attn, s_mask, t_ids, t_attn, t_mask = _align_student_teacher_batches(
                tokenizer, s_ids, s_attn, s_mask, t_ids, t_attn, t_mask
            )
            mask = (s_mask * t_mask).to(device)
            s_ids, s_attn = s_ids.to(device), s_attn.to(device)
            t_ids, t_attn = t_ids.to(device), t_attn.to(device)

            # Teacher (no grad)
            model.eval()
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=amp, dtype=amp_dt):
                t_logits = model(input_ids=t_ids, attention_mask=t_attn, use_cache=False).logits[:, :-1]

            # Student
            model.train()
            with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dt):
                s_logits = model(input_ids=s_ids, attention_mask=s_attn, use_cache=False).logits[:, :-1]
                kl = _distill_kl_loss(s_logits, t_logits, mask, T)
                loss = kl 

            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.optim.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)

            logs["loss"] += float(loss.detach())
            logs["kl"] += float(kl.detach())
            logs["steps"] += 1
    finally:
        if prev_cache is not None:
            model.config.use_cache = prev_cache

    if logs["steps"] > 0:
        for k in ("loss", "kl"):
            logs[k] /= logs["steps"]

    return get_peft_model_state_dict(model, adapter_name="default"), max(1, len(traces)), logs

# ---------------------------------------------------------------------------
# GRPO (kept byte-for-byte compatible with the original implementation)
# ---------------------------------------------------------------------------

def _scalar_reward(ex, y, cfg):
    return 1.0 if is_correct(y, ex, cfg.data.dataset, cfg) else 0.0


def _masked_logp(logits, input_ids, mask, per_token=False):
    target = input_ids[:, 1:]
    lp = torch.log_softmax(logits, dim=-1)
    tok_lp = lp.gather(-1, target.unsqueeze(-1)).squeeze(-1) * mask
    if per_token:
        return tok_lp
    return tok_lp.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


@torch.no_grad()
def _chunked_logps(model, adapter, ids, attn, mask, micro_bs, amp, amp_dt):
    model.eval()
    model.set_adapter(adapter)
    outs = []
    for j in range(0, ids.shape[0], max(1, micro_bs)):
        with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dt):
            o = model(input_ids=ids[j:j+micro_bs], attention_mask=attn[j:j+micro_bs],
                      use_cache=False)
            outs.append(_masked_logp(o.logits[:, :-1], ids[j:j+micro_bs],
                                     mask[j:j+micro_bs]))
    return torch.cat(outs) if outs else ids.new_empty((0,), dtype=torch.float32)


def client_grpo_update(model, tokenizer, global_adapter_state, client_examples,
                       cfg, feedback_provider, rng, round):
    """Group Relative Policy Optimisation baseline."""
    if "ref" not in getattr(model, "peft_config", {}):
        model.add_adapter("ref", list(model.peft_config.values())[0])

    set_peft_model_state_dict(model, global_adapter_state, adapter_name="default")
    set_peft_model_state_dict(model, global_adapter_state, adapter_name="ref")
    for n, p in model.named_parameters():
        if "lora_" in n and ".ref." in n:
            p.requires_grad = False
        elif "lora_" in n and ".default." in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    lr = _cosine_lr(round, cfg.fl.rounds, cfg.optim.lr, cfg.optim.lr_min,
                    cfg.optim.warmup_ratio, cfg.optim.use_warmup)
    opt = AdamW([p for p in model.parameters() if p.requires_grad],
                lr=lr, weight_decay=cfg.optim.weight_decay)

    G = max(1, cfg.grpo.num_generations)
    bsz = max(1, cfg.optim.batch_size)
    grad_accum = max(1, cfg.optim.grad_accum_steps)
    device = model.device
    amp = cfg.model.dtype in ("bf16", "fp16") and torch.cuda.is_available()
    amp_dt = torch.bfloat16 if cfg.model.dtype == "bf16" else torch.float16
    micro_bs = getattr(cfg.optim, "micro_batch_size", None) or 10**9

    gen_p = GenParams(max_new_tokens=cfg.gen.max_new_tokens_y0, do_sample=True,
                      temperature=max(0.7, cfg.gen.temperature),
                      top_p=cfg.gen.top_p, use_cache=True)

    prev_cache = getattr(model.config, "use_cache", None)
    if prev_cache is not None:
        model.config.use_cache = False

    logs = {"algo": "grpo", "steps": 0, "avg_reward": 0.0,
            "avg_kl": 0.0, "avg_pg_loss": 0.0}
    micro_steps = max(1, cfg.fl.local_train_steps) * grad_accum

    def _grpo_prompt(ex: Dict[str, Any]) -> str:
        """Return the prompt for a GRPO step, appending facts for StrategyQA."""
        p = ex["prompt"]
        if cfg.data.dataset == "strategy_qa":
            facts = list((ex.get("meta") or {}).get("facts") or [])
            if facts:
                facts_text = "\n".join(f"  - {f}" for f in facts)
                p = p + f"[RELEVANT FACTS]\n{facts_text}\n\n"
        return p

    try:
        for step in range(micro_steps):
            ex_batch = _sample(rng, client_examples, bsz)
            prompts = [_grpo_prompt(ex) for ex in ex_batch]

            # Generate G completions per prompt
            model.eval()
            model.set_adapter("default")
            prompts_rep = [p for p in prompts for _ in range(G)]
            ex_rep = [ex for ex in ex_batch for _ in range(G)]
            completions = generate_batch_hf(model, tokenizer, prompts_rep, gen_p)

            rewards = [_scalar_reward(ex, y, cfg) * cfg.grpo.reward_scale
                       for ex, y in zip(ex_rep, completions)]
            if cfg.grpo.reward_clip is not None:
                rc = float(cfg.grpo.reward_clip)
                rewards = [max(-rc, min(rc, r)) for r in rewards]

            # Compute advantages per group
            adv = []
            group_means = []
            for i in range(bsz):
                rs = rewards[i * G:(i + 1) * G]
                rm = sum(rs) / max(1, len(rs))
                group_means.append(rm)
                a = [r - rm for r in rs]
                if cfg.grpo.normalize_advantage and len(a) > 1:
                    m = sum(a) / len(a)
                    v = sum((x - m) ** 2 for x in a) / max(1, len(a) - 1)
                    s = v ** 0.5 if v > 0 else 1.0
                    a = [(x - m) / max(1e-8, s) for x in a]
                adv.extend(a)

            ids, attn, mask = _build_completion_batch(
                tokenizer, prompts_rep, completions, cfg.model.max_seq_len)
            ids, attn, mask = ids.to(device), attn.to(device), mask.to(device)

            old_lp = _chunked_logps(model, "default", ids, attn, mask,
                                    micro_bs, amp, amp_dt).detach()
            ref_lp = _chunked_logps(model, "ref", ids, attn, mask,
                                    micro_bs, amp, amp_dt).detach()

            # Policy gradient step
            model.train()
            model.set_adapter("default")
            new_chunks = []
            for j in range(0, ids.shape[0], micro_bs):
                with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dt):
                    o = model(input_ids=ids[j:j+micro_bs],
                              attention_mask=attn[j:j+micro_bs], use_cache=False)
                    new_chunks.append(_masked_logp(
                        o.logits[:, :-1], ids[j:j+micro_bs], mask[j:j+micro_bs]))
            new_lp = torch.cat(new_chunks) if new_chunks else torch.zeros(0, device=device)

            adv_t = torch.tensor(adv, device=device, dtype=new_lp.dtype)
            ratio = torch.exp(new_lp - old_lp)
            eps = float(cfg.grpo.clip_range)
            clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
            pg1, pg2 = ratio * adv_t, clipped * adv_t
            pg_obj = torch.where(adv_t >= 0, torch.minimum(pg1, pg2),
                                 torch.maximum(pg1, pg2))
            pg_loss = -pg_obj
            log_r = (new_lp - ref_lp).clamp(-5, 5)
            kl = torch.exp(log_r) - 1.0 - log_r
            loss = (pg_loss + cfg.grpo.beta_kl * kl).mean()

            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.optim.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)

            logs["steps"] += 1
            logs["avg_reward"] += sum(group_means) / max(1, len(group_means))
            logs["avg_kl"] += float(kl.detach().mean()) if kl.numel() else 0
            logs["avg_pg_loss"] += float(pg_loss.detach().mean()) if pg_loss.numel() else 0

    finally:
        if prev_cache is not None:
            model.config.use_cache = prev_cache

    updated = get_peft_model_state_dict(model, adapter_name="default")
    if logs["steps"] > 0:
        for k in ("avg_reward", "avg_kl", "avg_pg_loss"):
            logs[k] /= logs["steps"]

    return updated, len(client_examples), logs

# ---------------------------------------------------------------------------
# RLTF-SD  (Algorithm 1 from Song et al. 2026)
# ---------------------------------------------------------------------------
def client_rltf_sd_update(model, tokenizer, global_adapter_state, client_examples,
                           cfg, feedback_provider, rng, round):
    """RLTF-SD: RL from Text Feedback via Self-Distillation.

    Implements Algorithm 1 from:
      "Expanding the Capabilities of Reinforcement Learning via Text Feedback",
      Song, Chen, Tajwar, Munos, Pathak, Bagnell, Singh, Zanette (2026).
      arXiv:2602.02482v2

    ─── Overview ────────────────────────────────────────────────────────────
    Standard RL (GRPO) provides only a sparse scalar reward per rollout.
    RLTF-SD adds a *self-distillation* objective that uses the feedback-
    conditioned second-turn policy π(·|x1) as an implicit teacher for the
    single-turn policy π(·|x0).  By internalising the feedback signal during
    training, the model improves first-turn (test-time) performance even when
    no feedback is available at inference.

    ─── Interaction protocol (Section 2) ────────────────────────────────────
      Turn 0:  y0_i  ~ π(·| x0)            [N completions per prompt]
               r0_i   = R(x0, y0_i)         [binary correctness]
               c0_i  ~ M(x0, y0_i)          [text feedback]
      Turn 1:  x1_i   = f(x0, y0_i, c0_i)  [revision prompt]
               y1_i  ~ π(·| x1_i)
               r1_i   = R(x0, y1_i)

    ─── Advantage computation (Algorithm 1, lines 10-14) ────────────────────
      R_i    = r0_i + γ · r1_i                      [discounted return]
      b(0)   = (1/N) Σ r0_i                          [first-turn baseline]
      b(R)   = (1/N) Σ R_i                           [return baseline]
      b(1)   = (1/N) Σ r1_i                          [second-turn baseline]
      A_i    = r1_i - b(0)                           [SD advantage, Eq. 7]
      A_RL0_i = R_i  - b(R)                          [RL turn-0 advantage]
      A_RL1_i = r1_i - b(1)                          [RL turn-1 advantage]

    ─── Key design choices (Section 3) ──────────────────────────────────────
    (a) AWR-style distillation (no importance weighting):
          Equation (3): πref(·|x1) = π(·|x0)
        This removes the IS ratio π(y1|x0)/π(y1|x1), which can be
        exponentially large for long sequences (Table 3 empirical validation).
        Introduces mild bias but greatly reduces variance (Section 3.2).

    (b) First-turn mean baseline b(0) instead of second-turn mean (Section 3.1):
        Second-turn baseline → gradient-signal collapse when p1→1 (feedback
        makes the teacher reliable). Probability of non-zero update scales as
        1 - p1^N ≈ N(1-p1), vanishing as the teacher improves.
        First-turn baseline b(0) only vanishes when the *student itself* is
        already correct (b(0) → 1), which is the desired stopping condition.

    ─── Gradient computation (Algorithm 1, lines 15-17) ────────────────────
      g_SD  = (1/N) Σ_i  A_i    · ∇ log π(y1_i | x0)   [SD: y1 scored at x0!]
      g_RL  = (1/N) Σ_i [A_RL0_i · ∇ log π(y0_i | x0)
                        + A_RL1_i · ∇ log π(y1_i | x1_i)]
      θ ← OPT(θ, η, sd_coeff · g_SD + rl_coeff · g_RL)

    The SD gradient uses y1 scored under the *first-turn* context x0 (not
    x1).  This is the critical difference from naive multi-turn RL.

    Parameters (from cfg.rltf_sd):
      gamma           – discount for returns  (default 1.0)
      sd_coeff        – weight on g_SD        (default 0.1, Table 2)
      rl_coeff        – weight on g_RL        (default 1.0)
      early_termination – skip turn-1 if r0=1 (default True, Section 2)
    KL regularisation uses cfg.grpo.beta_kl (same as GRPO baseline).
    """
    if feedback_provider is None:
        raise ValueError("RLTF-SD requires a feedback provider (set feedback.provider).")

    # ── Set up a frozen reference adapter for KL regularisation ────────────
    if "ref" not in getattr(model, "peft_config", {}):
        model.add_adapter("ref", list(model.peft_config.values())[0])

    set_peft_model_state_dict(model, global_adapter_state, adapter_name="default")
    set_peft_model_state_dict(model, global_adapter_state, adapter_name="ref")
    for n, p in model.named_parameters():
        if "lora_" in n and ".ref." in n:
            p.requires_grad = False
        elif "lora_" in n and ".default." in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # ── Hyperparameters ─────────────────────────────────────────────────────
    rltf_cfg = cfg.rltf_sd
    N              = max(1, cfg.grpo.num_generations)   # group size (paper: N)
    gamma          = float(rltf_cfg.gamma)              # return discount
    sd_coeff       = float(rltf_cfg.sd_coeff)           # g_SD weight (Table 2: 0.1)
    rl_coeff       = float(rltf_cfg.rl_coeff)           # g_RL weight
    use_early_term = bool(rltf_cfg.early_termination)   # skip turn-1 when r0=1
    beta_kl        = float(cfg.grpo.beta_kl)            # KL penalty
    reward_scale   = float(cfg.grpo.reward_scale)
    ds             = cfg.data.dataset

    device  = model.device
    amp     = cfg.model.dtype in ("bf16", "fp16") and torch.cuda.is_available()
    amp_dt  = torch.bfloat16 if cfg.model.dtype == "bf16" else torch.float16
    micro_bs = getattr(cfg.optim, "micro_batch_size", None) or 10**9

    lr = _cosine_lr(round, cfg.fl.rounds, cfg.optim.lr, cfg.optim.lr_min,
                    cfg.optim.warmup_ratio, cfg.optim.use_warmup)
    opt = AdamW([p for p in model.parameters() if p.requires_grad],
                lr=lr, weight_decay=cfg.optim.weight_decay)

    grad_accum = max(1, cfg.optim.grad_accum_steps)
    bsz        = max(1, cfg.optim.batch_size)
    micro_steps = max(1, cfg.fl.local_train_steps) * grad_accum
    bsz_roll   = max(1, cfg.gen.rollout_batch_size)

    gen_p_y0 = GenParams(max_new_tokens=cfg.gen.max_new_tokens_y0,
                          do_sample=True,
                          temperature=max(0.7, cfg.gen.temperature),
                          top_p=cfg.gen.top_p, use_cache=True)
    gen_p_y1 = GenParams(max_new_tokens=cfg.gen.max_new_tokens_y1,
                          do_sample=True,
                          temperature=max(0.7, cfg.gen.temperature),
                          top_p=cfg.gen.top_p, use_cache=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: ROLLOUT — Algorithm 1, lines 3-9
    # Collect N two-turn trajectories per prompt from the current policy.
    # ══════════════════════════════════════════════════════════════════════════
    # Number of unique prompts: rollout_steps / N (keeps total rollouts ≈ budget)
    n_prompts  = min(len(client_examples),
                     max(1, cfg.fl.local_rollout_steps // N))
    rollout_ex = (rng.sample(client_examples, n_prompts)
                  if len(client_examples) >= n_prompts else client_examples)
    n_prompts  = len(rollout_ex)  # actual number after possible shortfall

    # Replicate each prompt N times for group rollout  (Algorithm 1, line 4)
    prompts_rep: List[str] = [ex["prompt"] for ex in rollout_ex for _ in range(N)]
    ex_rep:      List[Any] = [ex            for ex in rollout_ex for _ in range(N)]

    # — Turn 0: generate y0_i ~ π(·|x0) ————————————————————————————————————
    model.eval(); model.set_adapter("default")
    y0_all: List[str] = []
    for i in range(0, len(prompts_rep), bsz_roll):
        y0_all.extend(generate_batch_hf(
            model, tokenizer, prompts_rep[i:i + bsz_roll], gen_p_y0))

    # Scalar rewards r0_i = R(x0, y0_i)  (Algorithm 1, line 9)
    r0_all: List[float] = [
        reward_scale * (1.0 if _scalar_reward(ex, y, cfg) else 0.0)
        for ex, y in zip(ex_rep, y0_all)
    ]

    # — Early termination (Section 2): skip feedback when r0_i = 1 ——————————
    # When the first attempt is already correct there is nothing to critique;
    # for these samples y1 = y0 and r1 = r0 (episode terminates at turn 0).
    need_fb_set = set(range(len(y0_all)))
    if use_early_term:
        need_fb_set = {i for i, r in enumerate(r0_all) if r < reward_scale}

    # — Get text feedback c0_i from the feedback provider ——————————————————
    fb_max  = int(getattr(cfg.distill, "feedback_max_chars", 800))
    x1_all: List[str] = [prompts_rep[i] for i in range(len(y0_all))]  # default = x0
    y1_all: List[str] = list(y0_all)                                    # default = y0
    r1_all: List[float] = list(r0_all)                                  # default = r0

    if need_fb_set:
        fb_idx  = sorted(need_fb_set)
        fb_exs  = [ex_rep[i]   for i in fb_idx]
        fb_y0s  = [y0_all[i]   for i in fb_idx]
        fb_texts = feedback_provider.feedback_batch(
            fb_exs, fb_y0s, cfg.data, round)
        fb_texts = [_truncate_text(t, fb_max) for t in fb_texts]

        # Build revision prompts x1_i = f(x0, y0_i, c0_i) (Section 2, Eq. after r0)
        for idx, (i, fb) in enumerate(zip(fb_idx, fb_texts)):
            x1_all[i] = build_revision_prompt(
                ex_rep[i]["prompt"], y0_all[i], fb, ds)

        # — Turn 1: generate y1_i ~ π(·|x1_i) ──────────────────────────────
        active_x1 = [x1_all[i] for i in fb_idx]
        active_y1: List[str] = []
        for i in range(0, len(active_x1), bsz_roll):
            active_y1.extend(generate_batch_hf(
                model, tokenizer, active_x1[i:i + bsz_roll], gen_p_y1))

        for i, y1 in zip(fb_idx, active_y1):
            y1_all[i] = y1
            r1_all[i] = reward_scale * (1.0 if _scalar_reward(ex_rep[i], y1, cfg) else 0.0)

    print(f"[RLTF-SD] r0 mean={sum(r0_all)/max(1,len(r0_all)):.3f}  "
          f"r1 mean={sum(r1_all)/max(1,len(r1_all)):.3f}  "
          f"n_fb={len(need_fb_set)}/{len(y0_all)}", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: ADVANTAGE COMPUTATION — Algorithm 1, lines 10-14
    # Group samples by prompt (group size N), compute per-group baselines and
    # per-sample advantages for both the SD objective and the RL objective.
    # ══════════════════════════════════════════════════════════════════════════
    traces: List[Dict[str, Any]] = []
    for gi in range(n_prompts):
        s, e = gi * N, (gi + 1) * N
        grp_y0 = y0_all[s:e];  grp_y1 = y1_all[s:e]
        grp_x1 = x1_all[s:e]
        grp_r0 = r0_all[s:e];  grp_r1 = r1_all[s:e]
        x0     = rollout_ex[gi]["prompt"]
        Ni     = len(grp_r0)   # may be < N at end of list

        # Returns R_i = r0_i + γ · r1_i  (Algorithm 1, line 10)
        grp_R = [r0 + gamma * r1 for r0, r1 in zip(grp_r0, grp_r1)]

        # Baselines  (Algorithm 1, line 11)
        # b(0): first-turn mean — avoids gradient-signal collapse (Section 3.1)
        b0 = sum(grp_r0) / Ni
        bR = sum(grp_R)  / Ni
        b1 = sum(grp_r1) / Ni

        for j in range(Ni):
            # SD advantage: A_i = r1_i - b(0)  (Algorithm 1 line 12, Eq. 7)
            # Uses first-turn baseline so gradient is non-zero as long as the
            # *student* (not the teacher) is imperfect.
            A_sd  = grp_r1[j] - b0
            # RL advantages (Algorithm 1, lines 13-14)
            A_RL0 = grp_R[j]  - bR
            A_RL1 = grp_r1[j] - b1

            traces.append({
                "x0":   x0,
                "y0":   grp_y0[j],
                "x1":   grp_x1[j],   # revision prompt (= x0 if early-terminated)
                "y1":   grp_y1[j],   # second-turn output (= y0 if early-terminated)
                "r0":   grp_r0[j],
                "r1":   grp_r1[j],
                "A_sd": A_sd,        # for SD loss (Alg. 1 line 15)
                "A_RL0": A_RL0,      # for RL turn-0 (Alg. 1 line 16, term 1)
                "A_RL1": A_RL1,      # for RL turn-1 (Alg. 1 line 16, term 2)
            })

    if not traces:
        return (get_peft_model_state_dict(model, adapter_name="default"),
                max(1, len(client_examples)),
                {"algo": "rltf_sd", "steps": 0, "skipped": True,
                 "reason": "no_traces"})

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3: TRAINING — Algorithm 1, lines 15-17
    #
    # For each mini-batch compute three policy-gradient surrogate losses:
    #
    #   L_SD  = -mean_i [A_sd_i  · log π(y1_i | x0_i)]        [Alg.1 line 15]
    #   L_RL0 = -mean_i [A_RL0_i · log π(y0_i | x0_i)] + KL   [Alg.1 line 16]
    #   L_RL1 = -mean_i [A_RL1_i · log π(y1_i | x1_i)] + KL   [Alg.1 line 16]
    #
    # L_SD uses the AWR-style objective: y1 is scored under the FIRST-TURN
    # context x0 (not the revision context x1).  This is the key that
    # removes the high-variance importance-sampling ratio (Section 3.2).
    #
    # Combined: loss = sd_coeff · L_SD + rl_coeff · (L_RL0 + L_RL1)
    # ══════════════════════════════════════════════════════════════════════════
    prev_cache = getattr(model.config, "use_cache", None)
    if prev_cache is not None:
        model.config.use_cache = False

    logs = {
        "algo": "rltf_sd", "steps": 0,
        "sd_loss": 0.0, "rl0_loss": 0.0, "rl1_loss": 0.0,
        "avg_kl0": 0.0, "avg_kl1": 0.0,
        "avg_r0": sum(r0_all) / max(1, len(r0_all)),
        "avg_r1": sum(r1_all) / max(1, len(r1_all)),
        "n_traces": len(traces),
    }

    try:
        model.train(); model.set_adapter("default")

        for step in range(micro_steps):
            batch = _sample(rng, traces, bsz)
            B     = len(batch)

            x0_b  = [t["x0"]   for t in batch]
            y0_b  = [t["y0"]   for t in batch]
            x1_b  = [t["x1"]   for t in batch]
            y1_b  = [t["y1"]   for t in batch]

            A_sd  = torch.tensor([t["A_sd"]  for t in batch], device=device)
            A_RL0 = torch.tensor([t["A_RL0"] for t in batch], device=device)
            A_RL1 = torch.tensor([t["A_RL1"] for t in batch], device=device)

            # ── Build tokenised batches ──────────────────────────────────────
            # SD:  prefix=x0,  completion=y1   [y1 scored under x0, not x1!]
            # RL0: prefix=x0,  completion=y0
            # RL1: prefix=x1,  completion=y1
            ids_sd,  attn_sd,  mask_sd  = _build_completion_batch(
                tokenizer, x0_b, y1_b, cfg.model.max_seq_len)
            ids_rl0, attn_rl0, mask_rl0 = _build_completion_batch(
                tokenizer, x0_b, y0_b, cfg.model.max_seq_len)
            ids_rl1, attn_rl1, mask_rl1 = _build_completion_batch(
                tokenizer, x1_b, y1_b, cfg.model.max_seq_len)

            ids_sd,  attn_sd,  mask_sd  = ids_sd.to(device),  attn_sd.to(device),  mask_sd.to(device)
            ids_rl0, attn_rl0, mask_rl0 = ids_rl0.to(device), attn_rl0.to(device), mask_rl0.to(device)
            ids_rl1, attn_rl1, mask_rl1 = ids_rl1.to(device), attn_rl1.to(device), mask_rl1.to(device)

            # ── Reference log-probs for KL regularisation (no grad) ──────────
            ref_lp_y0 = _chunked_logps(
                model, "ref", ids_rl0, attn_rl0, mask_rl0, micro_bs, amp, amp_dt).detach()
            ref_lp_y1 = _chunked_logps(
                model, "ref", ids_rl1, attn_rl1, mask_rl1, micro_bs, amp, amp_dt).detach()

            # ── Reset to training mode after reference forward passes ────────
            model.train(); model.set_adapter("default")

            # ── L_SD: self-distillation loss (Algorithm 1, line 15) ──────────
            #
            #   g_SD = (1/N) Σ_i A_i · ∇ log π(y1_i | x0_i)
            #
            # AWR surrogate: L_SD = -mean(A_i · log π(y1_i | x0_i))
            # No importance weighting: πref(·|x1) = π(·|x0) [Section 3.2].
            # The advantage A_i = r1_i - b(0) has detach so gradients only
            # flow through log π(y1|x0).
            with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dt):
                logits_sd = model(input_ids=ids_sd, attention_mask=attn_sd,
                                  use_cache=False).logits[:, :-1]
                logp_y1_x0 = _masked_logp(logits_sd, ids_sd, mask_sd)
                sd_loss = -(A_sd.detach() * logp_y1_x0).mean()

            # ── L_RL0: RL turn-0 loss (Algorithm 1, line 16, first term) ─────
            #
            #   A_RL0_i · ∇ log π(y0_i | x0_i)  +  beta_kl · KL(π || πref)
            with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dt):
                logits_rl0 = model(input_ids=ids_rl0, attention_mask=attn_rl0,
                                   use_cache=False).logits[:, :-1]
                logp_y0    = _masked_logp(logits_rl0, ids_rl0, mask_rl0)
                pg0        = -(A_RL0.detach() * logp_y0)
                # KL ≈ exp(log r) - 1 - log r  (GRPO-style approximation)
                log_r0     = (logp_y0 - ref_lp_y0).clamp(-5, 5)
                kl0        = (torch.exp(log_r0) - 1.0 - log_r0)
                rl0_loss   = (pg0 + beta_kl * kl0).mean()

            # ── L_RL1: RL turn-1 loss (Algorithm 1, line 16, second term) ────
            #
            #   A_RL1_i · ∇ log π(y1_i | x1_i)  +  beta_kl · KL(π || πref)
            with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dt):
                logits_rl1 = model(input_ids=ids_rl1, attention_mask=attn_rl1,
                                   use_cache=False).logits[:, :-1]
                logp_y1_x1 = _masked_logp(logits_rl1, ids_rl1, mask_rl1)
                pg1        = -(A_RL1.detach() * logp_y1_x1)
                log_r1     = (logp_y1_x1 - ref_lp_y1).clamp(-5, 5)
                kl1        = (torch.exp(log_r1) - 1.0 - log_r1)
                rl1_loss   = (pg1 + beta_kl * kl1).mean()

            # ── Combined update  (Algorithm 1, line 17) ──────────────────────
            #   θ ← OPT(θ, η, sd_coeff · g_SD + rl_coeff · g_RL)
            loss = sd_coeff * sd_loss + rl_coeff * (rl0_loss + rl1_loss)

            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.optim.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)

            logs["steps"]    += 1
            logs["sd_loss"]  += float(sd_loss.detach())
            logs["rl0_loss"] += float(rl0_loss.detach())
            logs["rl1_loss"] += float(rl1_loss.detach())
            logs["avg_kl0"]  += float(kl0.detach().mean()) if kl0.numel() else 0.0
            logs["avg_kl1"]  += float(kl1.detach().mean()) if kl1.numel() else 0.0

    finally:
        if prev_cache is not None:
            model.config.use_cache = prev_cache

    if logs["steps"] > 0:
        for k in ("sd_loss", "rl0_loss", "rl1_loss", "avg_kl0", "avg_kl1"):
            logs[k] /= logs["steps"]

    return (get_peft_model_state_dict(model, adapter_name="default"),
            max(1, len(traces)), logs)

# ---------------------------------------------------------------------------
# Feedback SFT (no unlikehood or RL, just supervised fine-tuning on feedback-conditioned revisions)
# ---------------------------------------------------------------------------
def client_sft_feedback_update(
    model, tokenizer,
    global_adapter_state: Dict[str, torch.Tensor],
    client_examples: List[Dict[str, Any]],
    cfg: ExperimentConfig,
    feedback_provider,
    rng: random.Random,
    round: int,
) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, Any]]:
    """Run one round of feedback-guided SFT on a single client's data.

    Like client_spear_update but without unlikelihood training on lose traces.
    Generates y0, attempts feedback-guided revisions for incorrect outputs,
    then trains only with SFT on win traces.

    Returns (updated_adapter_state, effective_n_samples, logs_dict).
    """
    device = model.device
    max_seq = cfg.model.max_seq_len
    ds = cfg.data.dataset
    sp = cfg.spear

    # ── Load global adapter ──────────────────────────────────────────────
    set_peft_model_state_dict(model, global_adapter_state, adapter_name="default")

    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n and ".default." in n)

    ref_state = {k: v.detach().clone().to(device) for k, v in global_adapter_state.items()}

    # ── Phase 1: Rollout ─────────────────────────────────────────────────
    model.eval()
    model.set_adapter("default")

    n_rollout = min(len(client_examples), max(1, cfg.fl.local_rollout_steps))
    rollout_ex = (rng.sample(client_examples, n_rollout)
                  if len(client_examples) >= n_rollout else client_examples)

    bsz = max(1, cfg.gen.rollout_batch_size)
    gen0 = GenParams(max_new_tokens=cfg.gen.max_new_tokens_y0,
                     do_sample=cfg.gen.do_sample, temperature=cfg.gen.temperature,
                     top_p=cfg.gen.top_p, use_cache=True)
    gen1 = GenParams(max_new_tokens=cfg.gen.max_new_tokens_y1,
                     do_sample=cfg.gen.do_sample, temperature=cfg.gen.temperature,
                     top_p=cfg.gen.top_p, use_cache=True)

    prompts = [ex["prompt"] for ex in rollout_ex]
    y0_list: List[str] = []
    for chunk in _chunked(prompts, bsz):
        y0_list.extend(generate_batch_hf(model, tokenizer, chunk, gen0))

    y0_ok = [is_correct(y, ex, ds, cfg) for y, ex in zip(y0_list, rollout_ex)]

    win_traces: List[WinTrace] = []

    need_revision_wrong = []
    need_format_fix = []
    for i, (ex, y0, ok) in enumerate(zip(rollout_ex, y0_list, y0_ok)):
        if ok:
            if strict_win_ok(y0, ex, ds, cfg):
                win_traces.append(WinTrace(prompt=prompts[i], completion=y0))
            else:
                need_format_fix.append({"idx": i, "ex": ex, "prompt": prompts[i], "y0": y0})
        else:
            need_revision_wrong.append({"idx": i, "ex": ex, "prompt": prompts[i], "y0": y0})

    n_y0_ok = sum(y0_ok)
    print(f"\n📊 y0: {n_y0_ok}/{len(y0_list)} correct "
          f"({100 * n_y0_ok / max(1, len(y0_list)):.1f}%)", flush=True)

    # ------------------------------------------------------------------
    # 1) Format-fix pass (correct but misformatted y0) — no feedback needed
    # ------------------------------------------------------------------
    n_format_fixed = 0
    if need_format_fix:
        fmt_prompts = [
            build_format_fix_prompt(it["prompt"], it["y0"], ds)
            for it in need_format_fix
        ]
        y_fmt: List[str] = []
        for chunk in _chunked(fmt_prompts, bsz):
            y_fmt.extend(generate_batch_hf(model, tokenizer, chunk, gen1))

        for it, y in zip(need_format_fix, y_fmt):
            if strict_win_ok(y, it["ex"], ds, cfg):
                win_traces.append(WinTrace(prompt=it["prompt"], completion=y))
                n_format_fixed += 1

        print(f"📊 Format-fix: {n_format_fixed}/{len(need_format_fix)} successful", flush=True)

    # ------------------------------------------------------------------
    # 2) Feedback-guided revisions for incorrect y0 → y1
    # ------------------------------------------------------------------
    if need_revision_wrong and feedback_provider is not None:
        num_revs = max(1, int(getattr(cfg.gen, "num_revisions", 1)))

        for it in need_revision_wrong:
            it["cur"] = it["y0"]
            it["fixed"] = False
            it["best"] = None

        pending = need_revision_wrong
        n_revised_ok = 0

        for attempt in range(num_revs):
            if not pending:
                break
            fb_list = feedback_provider.feedback_batch(
                [it["ex"] for it in pending],
                [it["cur"] for it in pending],
                cfg.data, round
            )

            rev_prompts = [
                build_revision_prompt(it["prompt"], it["cur"], fb, ds)
                for it, fb in zip(pending, fb_list)
            ]

            y_list: List[str] = []
            for chunk in _chunked(rev_prompts, bsz):
                y_list.extend(generate_batch_hf(model, tokenizer, chunk, gen1))

            next_pending = []
            for it, y in zip(pending, y_list):
                if strict_win_ok(y, it["ex"], ds, cfg):
                    it["fixed"] = True
                    it["best"] = y
                    n_revised_ok += 1
                else:
                    it["cur"] = y
                    next_pending.append(it)
            pending = next_pending

        # Only commit successful revisions as win traces; discard the rest.
        for it in need_revision_wrong:
            if it.get("fixed") and it.get("best") is not None:
                win_traces.append(WinTrace(prompt=it["prompt"], completion=it["best"]))

        print(f"📊 Revisions: {n_revised_ok}/{len(need_revision_wrong)} successful "
              f"(attempts per sample: {num_revs})", flush=True)

    print(f"✅ {len(win_traces)} win traces for SFT", flush=True)

    _clear_gpu()

    # ── Phase 2: Training ────────────────────────────────────────────────
    if not win_traces:
        print("⚠️  No win traces. Skipping training.", flush=True)
        return (get_peft_model_state_dict(model, adapter_name="default"),
                max(1, n_rollout),
                {"win_loss": 0, "anchor_loss": 0, "steps": 0,
                 "wins": 0, "skipped": True})

    model.train()
    model.set_adapter("default")
    prev_cache = getattr(model.config, "use_cache", None)
    if prev_cache is not None:
        model.config.use_cache = False

    try:
        lr = _cosine_lr(round, cfg.fl.rounds, cfg.optim.lr, cfg.optim.lr_min,
                         cfg.optim.warmup_ratio, cfg.optim.use_warmup)
        train_params = [p for p in model.parameters() if p.requires_grad]
        opt = AdamW(train_params, lr=lr, weight_decay=cfg.optim.weight_decay)

        use_amp = cfg.model.dtype in ("bf16", "fp16") and torch.cuda.is_available()
        amp_dtype = torch.bfloat16 if cfg.model.dtype == "bf16" else torch.float16

        grad_accum = max(1, cfg.optim.grad_accum_steps)
        train_bsz = max(1, cfg.optim.batch_size)
        total_steps = max(1, cfg.fl.local_train_steps)
        micro_steps = total_steps * grad_accum

        lambda_win = sp.lambda_win
        lambda_anchor = sp.lambda_anchor

        logs = {"win_loss": 0.0, "anchor_loss": 0.0, "steps": 0,
                "wins": len(win_traces)}

        for step in range(micro_steps):
            loss = torch.tensor(0.0, device=device, requires_grad=True)

            # ── L_win: SFT on correct completions ────────────────────
            if win_traces and lambda_win > 0:
                batch = _sample(rng, win_traces, train_bsz)
                ids, attn, labels = _build_sft_batch(
                    tokenizer,
                    [t.prompt for t in batch],
                    [t.completion for t in batch],
                    max_seq,
                )
                ids = ids.to(device)
                attn = attn.to(device)
                labels = labels.to(device)

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits = model(input_ids=ids, attention_mask=attn,
                                   use_cache=False).logits
                    sft_loss = F.cross_entropy(
                        logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                        labels[:, 1:].contiguous().view(-1),
                        ignore_index=-100,
                    )

                loss = loss + lambda_win * sft_loss
                logs["win_loss"] += float(sft_loss.detach())
                del ids, attn, labels, logits

            # ── L_anchor: L2 proximity to reference adapter ─────────
            if lambda_anchor > 0:
                anchor = torch.tensor(0.0, device=device)
                current_state = get_peft_model_state_dict(model, adapter_name="default")
                for k, v in current_state.items():
                    if k in ref_state:
                        anchor = anchor + (v - ref_state[k]).pow(2).mean()
                loss = loss + lambda_anchor * anchor
                logs["anchor_loss"] += float(anchor.detach())

            # ── Backward + step ──────────────────────────────────────
            (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                if cfg.optim.max_grad_norm > 0:
                    gn = torch.nn.utils.clip_grad_norm_(
                        train_params, cfg.optim.max_grad_norm)
                    if torch.isnan(gn) or torch.isinf(gn):
                        opt.zero_grad(set_to_none=True)
                        continue
                opt.step()
                opt.zero_grad(set_to_none=True)

            logs["steps"] += 1

        # ── Finalise ─────────────────────────────────────────────────
        updated = get_peft_model_state_dict(model, adapter_name="default")

        ema = sp.ema_decay
        if ema > 0.0:
            for k in updated:
                if k in global_adapter_state:
                    updated[k] = (ema * global_adapter_state[k].to(updated[k].device)
                                  + (1.0 - ema) * updated[k])

        n_eff = max(1, len(win_traces))
        if logs["steps"] > 0:
            for key in ("win_loss", "anchor_loss"):
                logs[key] /= logs["steps"]

        return updated, n_eff, logs

    finally:
        if prev_cache is not None:
            model.config.use_cache = prev_cache
        _clear_gpu()


_DISPATCH = {
    "spear":    client_spear_update,
    "feedback_sft": client_sft_feedback_update,
    "grpo":     client_grpo_update,
    "opsd":     client_opsd_update,
    "rltf_sd":  client_rltf_sd_update,
}
# Algorithms that need a feedback provider
FEEDBACK_ALGORITHMS = {"spear", "rltf_sd", "feedback_sft"}
def get_client_update(cfg):
    name = getattr(cfg.algorithm, "name", "spear")
    if name not in _DISPATCH:
        raise ValueError(f"Unknown algorithm: {name}")
    return _DISPATCH[name]