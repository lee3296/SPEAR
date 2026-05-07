"""SPEAR: Self-Play Enhancement via Advantage-weighted Refinement.

A memory-efficient, single-generation online learning algorithm that uses
feedback-guided self-play to create natural contrastive training pairs.

Key differences from prior work:
  - vs GRPO:  Single generation per prompt (no multi-sampling group). 4-8x faster.
  - vs SFT:   Also pushes probability AWAY from confident wrong answers,
              not just toward correct ones.

Info:
  - L_win is MLE on filtered correct completions — consistent and convergent.
  - L_lose upper-bounds the probability of generating wrong answers.
"""
from __future__ import annotations

import gc
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from .config import ExperimentConfig
from .inference import GenParams, generate_batch_hf
from .prompting import format_prompt
from .extractors import (
    extract_mc_letter, extract_mc_letter_final, 
    extract_yes_no_final,
)


# ---------------------------------------------------------------------------
# Trace containers
# ---------------------------------------------------------------------------

@dataclass
class WinTrace:
    """A prompt→completion pair where the completion is correct."""
    prompt: str
    completion: str


@dataclass
class LoseTrace:
    """A prompt→completion pair where the completion is incorrect."""
    prompt: str
    completion: str
    confidence: float = 0.0  # higher = model was more confident yet wrong
    # When SPEARConfig.ul_answer_only is enabled, we restrict UL to the
    # answer-bearing suffix tokens (dataset-specific). This stores the number
    # of completion tokens (excluding EOS) to keep from the end.
    ul_suffix_tokens: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _cosine_lr(cur_round: int, total_rounds: int, peak_lr: float,
               min_lr: float, warmup_ratio: float, use_warmup: bool) -> float:
    if not use_warmup:
        return peak_lr
    warmup = int(warmup_ratio * total_rounds)
    if cur_round <= warmup:
        return peak_lr * cur_round / max(1, warmup)
    progress = (cur_round - warmup) / max(1, total_rounds - warmup)
    decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * decay


def _chunked(items: list, size: int):
    size = max(1, size)
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _sample(rng: random.Random, items: list, k: int) -> list:
    if not items or k <= 0:
        return []
    if len(items) >= k:
        return rng.sample(items, k)
    return [rng.choice(items) for _ in range(k)]


def _ul_answer_suffix(text: str, dataset: str) -> str:
    """Best-effort extraction of the *answer suffix* for UL masking.

    For arc_challenge / hellaswag: the full output is a single letter, so
    that letter is the suffix.

    For math_mcqa: the model generates reasoning followed by "Answer: X".
    We return only the final letter so UL penalises the wrong choice token,
    not the shared reasoning tokens.  ul_tail_tokens (typically 32) provides
    a broader fallback window that also covers the "Answer: X" declaration.

    Whitespace notes:
      - The regex uses \s* so "Answer:A", "Answer: A", "Answer:  A" all match.
      - The returned letter is always stripped to a bare letter (no space) so
        the tokeniser lookup in _annotate_ul_suffix_tokens is deterministic.
        (The _ul_format_protect_ids set handles protecting "Answer:" itself.)

    Returns "" when no reliable suffix is found.
    """
    if not text:
        return ""
    s = str(text)

    if dataset == "math_mcqa":
        # Match "Answer: X" / "Answer:X" / "Answer - X" etc. (case-insensitive).
        # \s* handles any number of spaces between the colon and the letter.
        hits = list(re.finditer(
            r"(?i)\bAnswer\s*[:\-]\s*([A-D])\b", s
        ))
        if hits:
            return hits[-1].group(1).strip().upper()
        # Fallback: use extract_mc_letter_final (respects last-declared-answer
        # semantics) rather than bare extract_mc_letter, which takes the first.
        pred = extract_mc_letter_final(s)
        return pred.strip().upper() if pred else ""

    if dataset in ("arc_challenge", "hellaswag"):
        # For bare-letter outputs the whole response IS the answer.
        pred = extract_mc_letter(s)
        return pred.strip().upper() if pred else ""

    if dataset == "strategy_qa":
        # Match "Answer: yes" / "Answer: no" (case-insensitive).
        hits = list(re.finditer(
            r"(?i)\bAnswer\s*[:\-]\s*(yes|no)\b", s
        ))
        if hits:
            return hits[-1].group(1).strip().lower()
        # Fallback: use extract_yes_no_final so intermediate yes/no mentions
        # in the reasoning chain don't shadow the final declaration.
        pred = extract_yes_no_final(s)
        return pred.strip().lower() if pred else ""

    return ""

def _annotate_ul_suffix_tokens(tokenizer, lose_traces: List[LoseTrace], dataset: str, *,
                               max_tokens_cap: int = 0) -> None:
    """Populate LoseTrace.ul_suffix_tokens for answer-only UL.

    We keep this lightweight: tokenize only the extracted suffixes (batch tokenization).
    """
    if not lose_traces:
        return
    suffixes = [_ul_answer_suffix(t.completion, dataset) for t in lose_traces]
    enc = tokenizer(suffixes, add_special_tokens=False, truncation=False)
    ids_list = enc.get("input_ids") or []

    for t, ids, suf in zip(lose_traces, ids_list, suffixes):
        n = len(ids) if suf else 0
        if max_tokens_cap and n > max_tokens_cap:
            n = max_tokens_cap
        t.ul_suffix_tokens = int(n)



def _format_revision_instructions(dataset: str) -> str:
    if dataset == "math_mcqa":
        return (
            'Reason through the problem step by step, then end your response with '
            'exactly: "Answer: X" where X is A, B, C, or D.'
        )
    if dataset in ("arc_challenge", "hellaswag"):
        return 'Answer with a single letter choice like "A", "B", "C", or "D".'
    if dataset == "strategy_qa":
        return (
            'Reason through the question step by step, then end your response with '
            'exactly: "Answer: yes" or "Answer: no".'
        )
    return "Provide a concise, correct final answer."

def build_revision_prompt(task_prompt: str, prev_answer: str, feedback: str, dataset: str) -> str:
    """Build a high-clarity revision prompt that encourages clean rewrites."""
    instr = _format_revision_instructions(dataset)
    return (
        "[TASK]\n"
        f"{task_prompt}\n\n"
        "[YOUR PREVIOUS ANSWER — DO NOT COPY]\n"
        f"{prev_answer}\n\n"
        "[FEEDBACK]\n"
        f"{feedback}\n\n"
        "[INSTRUCTIONS]\n"
        "Write the solution as if this is your first attempt.\n"
        "Use FEEDBACK internally, but do NOT mention or allude to them.\n"
        "No preamble, no meta-commentary.\n"
        "The first token must be the start of the solution (not 'Sure', not 'Let me', not 'Based on').\n"
        "Forbidden (case-insensitive): let me redo, let me try again, based on the feedback.\n"
        "Rewrite the entire solution from scratch.\n"
        "Keep it concise and correct.\n"
        f"{instr}\n\n"
        "[CORRECTED SOLUTION]\n"
    )


# ---------------------------------------------------------------------------
# Strict-format validation (used to filter *wins* so we don't SFT on lucky /
# misformatted answers that pollute the training distribution).
# ---------------------------------------------------------------------------
def strict_win_ok(answer: str, example: Dict[str, Any], dataset: str,
                  cfg: Optional[ExperimentConfig] = None) -> bool:
    """Correct AND strict-format."""
    if not is_correct(answer, example, dataset, cfg):
        return False

    meta = example.get("meta", {}) or {}

    # For math_mcqa and other MC tasks: no extra strict-format requirement.
    # Other datasets: no extra strict-format requirement.
    return True


def build_format_fix_prompt(task_prompt: str, prev_answer: str, dataset: str) -> str:
    """Prompt to *only* fix answer formatting for already-correct solutions."""
    instr = _format_revision_instructions(dataset)
    return (
        "[TASK]\n"
        f"{task_prompt}\n\n"
        "[YOUR PREVIOUS ANSWER]\n"
        f"{prev_answer}\n\n"
        "[INSTRUCTIONS]\n"
        "Your solution may be correct, but the FINAL-ANSWER FORMAT is invalid.\n"
        "Keep the solution/answer the same, but rewrite it cleanly so the last line matches the required format EXACTLY.\n"
        "No preamble, no meta-commentary.\n"
        f"{instr}\n\n"
        "[FORMATTED SOLUTION]\n"
    )

# ---------------------------------------------------------------------------
# Correctness oracle (binary signal only — answer never enters the context)
# ---------------------------------------------------------------------------

def is_correct(answer: str, example: Dict[str, Any], dataset: str,
               cfg: Optional[ExperimentConfig] = None) -> bool:
    """Return True if *answer* is correct for *example*.

    This is a binary evaluation oracle: it tells us right/wrong but never
    injects the gold answer into the model's prompt or context.
    """
    if not answer or not answer.strip():
        return False
    meta = example.get("meta", {})

    if dataset in ("arc_challenge", "hellaswag"):
        gold = (meta.get("reference") or "").upper()
        pred = (extract_mc_letter(answer) or "").upper()
        return bool(gold) and pred == gold

    if dataset == "math_mcqa":
        # Use last-declared-answer extraction: the model generates reasoning
        # then "Answer: X", so we want the final declared letter, not the
        # first letter mentioned in the chain.
        gold = (meta.get("reference") or "").upper()
        pred = (extract_mc_letter_final(answer) or "").upper()
        return bool(gold) and pred == gold

    if dataset == "strategy_qa":
        gold = (meta.get("reference") or meta.get("answer") or "").strip().lower()
        pred = (extract_yes_no_final(answer) or "").strip().lower()
        return bool(gold) and pred == gold

    return False


# ---------------------------------------------------------------------------
# Tokenisation helper
# ---------------------------------------------------------------------------

def _build_sft_batch(tokenizer, prompts_raw: List[str], completions: List[str],
                     max_seq_len: int, add_eos: bool = True):
    """Build (input_ids, attention_mask, labels) for causal-LM SFT.

    Labels are -100 for all prompt tokens and real token-ids for completion
    tokens, so cross-entropy loss only applies to the completion.
    """
    B = len(prompts_raw)
    assert B == len(completions) and B > 0

    prompts = [format_prompt(tokenizer, p) for p in prompts_raw]
    p_enc = tokenizer(prompts, add_special_tokens=False, truncation=False)
    c_enc = tokenizer(completions, add_special_tokens=False, truncation=False)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id or 0

    seqs, labs, max_len = [], [], 0
    for p_ids, c_ids in zip(p_enc["input_ids"], c_enc["input_ids"]):
        c = list(c_ids)
        if add_eos and eos_id is not None:
            c.append(eos_id)
        # Truncate completion first, then prefix
        max_c = max_seq_len - 1
        if len(c) > max_c:
            c = c[:max_c]
        max_p = max_seq_len - len(c)
        p = list(p_ids)[-max_p:] if len(p_ids) > max_p else list(p_ids)

        seq = p + c
        lab = [-100] * len(p) + c  # loss only on completion tokens
        seqs.append(seq)
        labs.append(lab)
        max_len = max(max_len, len(seq))

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)

    for i, (seq, lab) in enumerate(zip(seqs, labs)):
        L = len(seq)
        input_ids[i, :L] = torch.tensor(seq, dtype=torch.long)
        attn[i, :L] = 1
        labels[i, :L] = torch.tensor(lab, dtype=torch.long)

    return input_ids, attn, labels

# ---------------------------------------------------------------------------
# Main SPEAR client update
# ---------------------------------------------------------------------------

def client_spear_update(
    model, tokenizer,
    global_adapter_state: Dict[str, torch.Tensor],
    client_examples: List[Dict[str, Any]],
    cfg: ExperimentConfig,
    feedback_provider,
    rng: random.Random,
    round: int,
) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, Any]]:
    """Run one round of SPEAR on a single client's data.
    Returns (updated_adapter_state, effective_n_samples, logs_dict).
    """
    device = model.device
    max_seq = cfg.model.max_seq_len
    ds = cfg.data.dataset
    sp = cfg.spear

    # ── Load global adapter ──────────────────────────────────────────────
    set_peft_model_state_dict(model, global_adapter_state, adapter_name="default")

    # Freeze all but LoRA-default
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n and ".default." in n)

    # Reference state for anchor loss
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

    # Generate y0
    prompts = [ex["prompt"] for ex in rollout_ex]
    y0_list: List[str] = []
    for chunk in _chunked(prompts, bsz):
        y0_list.extend(generate_batch_hf(model, tokenizer, chunk, gen0))

    y0_ok = [is_correct(y, ex, ds, cfg) for y, ex in zip(y0_list, rollout_ex)]

    win_traces: List[WinTrace] = []
    lose_traces: List[LoseTrace] = []

    # Collect strict-format wins from y0.
    # If y0 is correct but misformatted, we try a lightweight format-fix rewrite
    # (no external feedback needed) and only then add it as a win.
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
            # If the rewrite fails, we skip it entirely (do NOT add a lose trace)
            # because the original y0 was correct.

        print(f"📊 Format-fix: {n_format_fixed}/{len(need_format_fix)} successful", flush=True)

    # ------------------------------------------------------------------
    # 2) Feedback-guided revisions for incorrect y0 → y1
    # ------------------------------------------------------------------
    if need_revision_wrong and feedback_provider is not None:
        num_revs = max(1, int(getattr(cfg.gen, "num_revisions", 1)))

        # Track per-item revision state
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

            # Generate one rewrite for each pending item in this attempt
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

        # Commit traces
        for it in need_revision_wrong:
            if it.get("fixed") and it.get("best") is not None:
                win_traces.append(WinTrace(prompt=it["prompt"], completion=it["best"]))
                # Mark the original y0 as a lose trace (confidently wrong)
                lose_traces.append(LoseTrace(
                    prompt=it["prompt"], completion=it["y0"], confidence=1.0))
            else:
                # Still wrong after all revisions → y0 is a lose trace
                lose_traces.append(LoseTrace(
                    prompt=it["prompt"], completion=it["y0"], confidence=0.5))

        print(f"📊 Revisions: {n_revised_ok}/{len(need_revision_wrong)} successful "
              f"(attempts per sample: {num_revs})", flush=True)

    else:
        # No feedback provider — all incorrect y0 become lose traces
        for item in need_revision_wrong:
            lose_traces.append(LoseTrace(
                prompt=item["prompt"], completion=item["y0"], confidence=0.5))

    print(f"✅ {len(win_traces)} win | ❌ {len(lose_traces)} lose", flush=True)

    _clear_gpu()

    # ── Phase 2: Training ────────────────────────────────────────────────
    # Guard: if there are no wins, skip training entirely.
    # Pure UL (no SFT) only suppresses tokens with nothing to reinforce,
    # which degrades the model monotonically and triggers a collapse spiral
    # (worse model → more loses → more UL → worse model).
    if not win_traces:
        print("⚠️  No win traces. Skipping training to avoid pure-UL collapse.",
              flush=True)
        return (get_peft_model_state_dict(model, adapter_name="default"),
                max(1, n_rollout),
                {"win_loss": 0, "lose_loss": 0, "anchor_loss": 0, "steps": 0,
                 "wins": 0, "losses": len(lose_traces), "skipped": True})

    # Cap lose traces to prevent UL from overwhelming the SFT signal.
    # When the model is struggling (low y0 accuracy), lose traces accumulate
    # rapidly while wins stay scarce.  Without a cap, each training step
    # applies equal UL and SFT gradient pressure, but the lose pool is much
    # larger so UL sees more unique data per effective epoch.
    # Subsampling loses to max_lose_ratio * wins keeps the two signals balanced.
    max_lose = int(sp.max_lose_ratio * len(win_traces))
    if len(lose_traces) > max_lose:
        lose_traces = rng.sample(lose_traces, max_lose)
        print(f"⚠️  Capped lose traces to {max_lose} "
              f"({sp.max_lose_ratio:.1f}× {len(win_traces)} wins)", flush=True)

    # If requested, precompute the answer-suffix token lengths for UL masking.
    # Do this *after* capping lose traces to keep overhead negligible.
    if getattr(sp, "ul_answer_only", False) and ds in ("arc_challenge", "hellaswag", "math_mcqa"):
        _annotate_ul_suffix_tokens(
            tokenizer,
            lose_traces,
            ds,
            max_tokens_cap=int(getattr(sp, "ul_tail_tokens", 0) or 0),
        )

    total_traces = len(win_traces) + len(lose_traces)
    if total_traces == 0:
        print("⚠️  No traces. Skipping training.", flush=True)
        return (get_peft_model_state_dict(model, adapter_name="default"),
                max(1, n_rollout),
                {"win_loss": 0, "lose_loss": 0, "anchor_loss": 0, "steps": 0,
                 "wins": 0, "losses": 0, "skipped": True})

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
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

        # Training parameters from config
        lambda_win = sp.lambda_win
        lambda_lose = sp.lambda_lose
        lambda_anchor = sp.lambda_anchor
        ul_margin = sp.unlikelihood_margin

        # ── Precompute protected token IDs (once, before the training loop) ──
        #
        # For MATH we *always* protect the format-structural delimiter
        # tokens (e.g., #### and \boxed{…}).  These tokens are required by ALL correct
        # answers, so letting UL suppress them causes rapid capability collapse
        # regardless of whether ul_answer_only is on.  The ul_answer_only flag
        # already handles this for the answer-only path by excluding delimiters
        # from the suffix; here we extend the same protection to the plain
        # ul_tail_tokens path so both modes are safe.
        #
        # If ul_skip_math_ops is True we additionally protect mathematical
        # operators and punctuation (+, -, =, (, ), \frac, etc.).  These tokens
        # are heavily shared across correct and incorrect completions, so
        # penalising them is more likely to damage general arithmetic capability
        # than to teach the model to avoid the specific wrong answer value.
        _ul_format_protect_ids: set = set()
        _ul_op_protect_ids: set = set()

        # math_mcqa: protect all tokenisations of the "Answer:" marker.
        # The model ends every completion with "Answer: X", so "Answer:" is a
        # structural format token shared by ALL completions (correct and wrong).
        # Applying UL to it would suppress the model's ability to produce the
        # required output format — we exclude it entirely.
        # We encode multiple whitespace variants because BPE tokenisers often
        # represent " Answer" (space-prefixed) as a different token to "Answer".
        if ds == "math_mcqa" or ds == 'strategy_qa':
            _answer_marker_variants = [
                "Answer:", " Answer:", "Answer: ", " Answer: ",
                "Answer",  " Answer",  "answer:",  " answer:",
                "answer",  " answer",
                # Also protect the bare ":" in case the tokeniser splits it
                # (less critical but harmless — ":" appears in the tail window)
            ]
            for _av in _answer_marker_variants:
                try:
                    _ul_format_protect_ids.update(
                        tokenizer.encode(_av, add_special_tokens=False)
                    )
                except Exception:
                    pass

        if getattr(sp, "ul_skip_math_ops", False) and ds == "math_mcqa":
            # math_mcqa reasoning chains contain real math — protect operators
            # Operators that are structurally necessary for ALL math expressions.
            # Suppressing these does not teach the model to avoid the wrong
            # answer value — it only degrades arithmetic formatting.
            _op_strings = [
                # basic ascii operators / punctuation
                "+", "-", "*", "/", "=", "^", "%", "!", "_",
                "<", ">", "<=", ">=", ",", ".", ":", ";",

                # delimiters (ascii)
                "(", ")", "[", "]", "{", "}", "|",

                # LaTeX literal braces + named braces
                "\\{", "\\}", "\\lbrace", "\\rbrace",

                # common LaTeX arithmetic / structure
                "\\frac", "\\dfrac", "\\tfrac",
                "\\sqrt",
                "\\times", "\\div", "\\cdot",
                "\\pm", "\\mp",
                "\\left", "\\right",
                "\\left(", "\\right)", "\\left[", "\\right]", "\\left\\{", "\\right\\}",
                "\\left|", "\\right|",
                "\\lvert", "\\rvert",

                # floor / ceil
                "\\lfloor", "\\rfloor", "\\lceil", "\\rceil",

                # relations / comparisons
                "\\le", "\\ge", "\\neq",
                "\\approx", "\\sim", "\\equiv", "\\propto",
                "\\to", "\\Rightarrow", "\\Leftarrow", "\\iff",

                # set / logic
                "\\in", "\\notin",
                "\\subset", "\\subseteq",
                "\\cup", "\\cap", "\\setminus",
                "\\forall", "\\exists",
                "\\land", "\\lor", "\\neg",

                # combinatorics
                "\\binom", "\\choose",

                # ellipses
                "\\ldots", "\\cdots", "\\dots",

                # constants / common symbols
                "\\pi", "\\infty",

                # common functions
                "\\sin", "\\cos", "\\tan",
                "\\log", "\\ln",

                # common Greek letters (frequent in MATH)
                "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\varepsilon",
                "\\zeta", "\\eta", "\\theta", "\\vartheta",
                "\\iota", "\\kappa", "\\lambda", "\\mu", "\\nu",
                "\\xi", "\\omicron", "\\rho", "\\varrho",
                "\\sigma", "\\varsigma", "\\tau",
                "\\upsilon", "\\phi", "\\varphi",
                "\\chi", "\\psi", "\\omega",

                # uppercase Greek (sometimes appears)
                "\\Gamma", "\\Delta", "\\Theta", "\\Lambda", "\\Xi",
                "\\Pi", "\\Sigma", "\\Upsilon", "\\Phi", "\\Psi", "\\Omega",
            ]
            for _os in _op_strings:
                try:
                    _ul_op_protect_ids.update(
                        tokenizer.encode(_os, add_special_tokens=False)
                    )
                except Exception:
                    pass

        # Union of all token IDs that must never receive UL gradient.
        _ul_all_protect_ids: set = _ul_format_protect_ids | _ul_op_protect_ids

        logs = {"win_loss": 0.0, "lose_loss": 0.0, "anchor_loss": 0.0,
                "steps": 0, "wins": len(win_traces), "losses": len(lose_traces)}

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
                    # Standard cross-entropy on completion tokens
                    sft_loss = F.cross_entropy(
                        logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                        labels[:, 1:].contiguous().view(-1),
                        ignore_index=-100,
                    )

                loss = loss + lambda_win * sft_loss
                logs["win_loss"] += float(sft_loss.detach())
                del ids, attn, labels, logits

            # ── L_lose: Unlikelihood on incorrect completions ────────
            if lose_traces and lambda_lose > 0:
                batch_l = _sample(rng, lose_traces, train_bsz)
                ids, attn, labels = _build_sft_batch(
                    tokenizer,
                    [t.prompt for t in batch_l],
                    [t.completion for t in batch_l],
                    max_seq,
                )
                ids = ids.to(device)
                attn = attn.to(device)
                labels = labels.to(device)

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits = model(input_ids=ids, attention_mask=attn,
                                   use_cache=False).logits[:, :-1]
                    targets = labels[:, 1:].contiguous()

                    # Build base position mask: only supervised (non -100) positions.
                    mask = (targets != -100).float()

                    # ── UL masking strategy ─────────────────────────────
                    # Default: optionally restrict to last N tokens via ul_tail_tokens.
                    # Answer-only mode: restrict UL to the answer-bearing
                    ul_tail = int(getattr(sp, "ul_tail_tokens", 0) or 0)
                    use_answer_only = bool(getattr(sp, "ul_answer_only", False)) and ds in ("arc_challenge", "hellaswag", "math_mcqa")

                    # Always exclude EOS from UL
                    _eos_id = tokenizer.eos_token_id
                    if _eos_id is not None:
                        mask = mask * (targets != int(_eos_id)).float()

                    if use_answer_only:
                        # Restrict UL to the per-trace answer suffix tokens.
                        # The suffix was pre-annotated by _annotate_ul_suffix_tokens
                        # to contain ONLY the answer value (e.g. "32" or the content
                        # inside \boxed{}), excluding the delimiter tokens (#### /
                        # \boxed{…}) that are shared with all correct answers.
                        tail_mask = torch.zeros_like(mask)
                        for b_idx, tr in enumerate(batch_l):
                            sup_pos = mask[b_idx].nonzero(as_tuple=False).squeeze(-1)
                            if sup_pos.numel() == 0:
                                continue
                            k = int(getattr(tr, "ul_suffix_tokens", 0) or 0)
                            if k <= 0:
                                # Safeguard fallback: last N completion tokens.
                                k = ul_tail
                            if k > 0:
                                keep = sup_pos[-min(k, sup_pos.numel()):]
                                tail_mask[b_idx, keep] = 1.0
                            else:
                                # No restriction requested → keep all supervised positions.
                                tail_mask[b_idx, sup_pos] = 1.0
                        mask = tail_mask

                    else:
                        # ul_tail_tokens > 0: restrict UL loss to the last N tokens
                        # of each completion.
                        if ul_tail > 0:
                            tail_mask = torch.zeros_like(mask)
                            for b_idx in range(mask.shape[0]):
                                sup_pos = mask[b_idx].nonzero(as_tuple=False).squeeze(-1)
                                if sup_pos.numel() > 0:
                                    keep = sup_pos[-min(ul_tail, sup_pos.numel()):]
                                    tail_mask[b_idx, keep] = 1.0
                            mask = tail_mask

                    # ── Protected-token masking ──────────────────────────
                    # Zero out any remaining mask positions whose target token
                    # belongs to the protected set (format delimiters and,
                    # optionally, math operators).  This applies to BOTH the
                    # answer-only and the tail-N paths:
                    #
                    #   • If ul_skip_math_ops is True, _ul_op_protect_ids also
                    #     covers +, -, =, (, ), \frac, etc.
                    if _ul_all_protect_ids and targets.numel() > 0:
                        protect_mask = torch.zeros_like(targets, dtype=torch.bool)
                        for _pid in _ul_all_protect_ids:
                            protect_mask = protect_mask | (targets == _pid)
                        # Zero out protected positions.
                        mask = mask * (~protect_mask).float()

                    # Unlikelihood: for each masked position,
                    # compute -log(1 - p(token)) to push probability away from
                    # the wrong completion tokens.
                    log_probs = F.log_softmax(logits, dim=-1)
                    token_lp = log_probs.gather(
                        -1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                    # Unlikelihood: -log(1 - p(token)), with margin clamp
                    p_token = token_lp.exp().clamp(max=1.0 - 1e-7)
                    ul = -torch.log(1.0 - p_token + 1e-8)
                    # Only penalise positions above the margin
                    ul = torch.where(p_token > ul_margin, ul, torch.zeros_like(ul))
                    ul = (ul * mask).sum() / mask.sum().clamp(min=1.0)

                    # Weight by confidence of each trace
                    conf = torch.tensor([t.confidence for t in batch_l],
                                        device=device, dtype=ul.dtype).mean()
                    ul_loss = ul * conf

                loss = loss + lambda_lose * ul_loss
                logs["lose_loss"] += float(ul_loss.detach())
                del ids, attn, labels, logits, log_probs

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

        # Optional EMA smoothing
        ema = sp.ema_decay
        if ema > 0.0:
            for k in updated:
                if k in global_adapter_state:
                    updated[k] = (ema * global_adapter_state[k].to(updated[k].device)
                                  + (1.0 - ema) * updated[k])

        # Weight this client's update in FedAvg by win count, not total traces.
        n_eff = max(1, len(win_traces))
        if logs["steps"] > 0:
            for key in ("win_loss", "lose_loss", "anchor_loss"):
                logs[key] /= logs["steps"]

        return updated, n_eff, logs

    finally:
        if prev_cache is not None:
            model.config.use_cache = prev_cache
        _clear_gpu()