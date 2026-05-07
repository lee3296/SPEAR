"""Evaluation metrics for all supported datasets.

MC tasks (arc_challenge, hellaswag) use log-likelihood scoring:
  For each answer choice, we compute log P(choice_letter | prompt) and pick
  the argmax. This matches the methodology used by lm-evaluation-harness and
  Open LLM Leaderboard, making results directly comparable.

math_mcqa and all other tasks use generative evaluation:
  The model generates free text; answer extraction pulls the final declared
  letter (or label/number) from the output.
"""
from __future__ import annotations

from typing import Any, Dict, List
import torch
from peft import set_peft_model_state_dict
from tqdm import tqdm

from .config import ExperimentConfig
from .extractors import (
    extract_mc_letter_final,
    extract_yes_no_final,
)
from .inference import GenParams, generate_batch_hf
from .prompting import format_prompt


# ---------------------------------------------------------------------------
# Log-likelihood scoring for multiple-choice tasks
# ---------------------------------------------------------------------------

MC_CHOICE_LABELS = ["A", "B", "C", "D"]


@torch.no_grad()
def _score_mc_logprobs_batched(
    model,
    tokenizer,
    prompts: List[str],
    num_choices: int = 4,
    batch_size: int = 8,
) -> List[int]:
    """Score multiple-choice questions via log-likelihood and return predicted indices.

    For each example the prompt already ends with "Answer:" (or similar). We
    append each candidate letter in turn and pick the letter whose
    log-probability under the model is highest.

    This matches the methodology used by lm-evaluation-harness and the Open
    LLM Leaderboard, so results are directly comparable to published numbers.

    IMPORTANT — format consistency:
    Training uses format_prompt() (which applies the chat template for
    instruct/chat models). To avoid a training-eval mismatch we apply the
    same format here.  For base models format_prompt() is a no-op, so the
    behaviour is unchanged.  For chat/instruct models (e.g. Llama-3.2-3B-
    Instruct) the raw prompt is out-of-distribution after chat-template
    fine-tuning, causing near-random log-prob scores regardless of training.

    Args:
        model:        A PEFT-wrapped causal LM in eval mode.
        tokenizer:    The matching tokenizer.
        prompts:      List of RAW prompts (not yet formatted), each ending
                      just before the answer token.
        num_choices:  Number of answer choices per question (default 4: A-D).
        batch_size:   Number of (prompt, choice) pairs per forward pass.

    Returns:
        List of predicted choice indices (0-based), one per input prompt.
    """
    device = next(model.parameters()).device
    labels = MC_CHOICE_LABELS[:num_choices]

    # Apply the same prompt formatting used during training so that the model
    # sees the choice letter in the same positional context it was trained on.
    # For base models is_chat_model() is False and format_prompt() is a no-op.
    # For instruct models the chat template is applied and add_generation_prompt
    # ensures the letter appears as the first token of the assistant turn.
    formatted_prompts = [format_prompt(tokenizer, p) for p in prompts]

    # --- Robust log-likelihood scoring (multi-token continuations) ---------
    # Some tokenizers do not represent the choice as a single token (e.g.
    # whitespace + letter).  Score the full continuation token sequence like
    # lm-evaluation-harness does.

    joiners = ["" if (fp and fp[-1].isspace()) else " " for fp in formatted_prompts]

    # Continuations include any separator needed to ensure the choice is in the
    # same context the model would generate it.
    cont_texts: List[str] = []
    for j, fp in enumerate(formatted_prompts):
        for lab in labels:
            cont_texts.append(joiners[j] + lab)

    cont_ids_list = tokenizer(cont_texts, add_special_tokens=False).input_ids
    max_cont_len = max((len(x) for x in cont_ids_list), default=1)

    # Leave room for the continuation so it is never truncated off the end.
    max_prompt_len = max(1, int(tokenizer.model_max_length) - int(max_cont_len))
    prompt_enc = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
    )
    prompt_ids = prompt_enc["input_ids"]
    prompt_attn = prompt_enc["attention_mask"]
    prompt_pad_len = int(prompt_ids.shape[1])

    # Build all (prompt_tokens + continuation_tokens) sequences.
    pad_id = int(tokenizer.pad_token_id)
    all_input_ids: List[torch.Tensor] = []
    all_attn: List[torch.Tensor] = []
    cont_lens: List[int] = []

    k = 0
    for i in range(len(prompts)):
        base_ids = prompt_ids[i]
        base_attn = prompt_attn[i]
        for _ in range(num_choices):
            cids = list(cont_ids_list[k])
            k += 1
            cont_lens.append(len(cids))
            pad_len = max_cont_len - len(cids)
            cont = torch.tensor(cids + [pad_id] * pad_len, dtype=torch.long)
            cont_attn = torch.tensor([1] * len(cids) + [0] * pad_len, dtype=torch.long)
            all_input_ids.append(torch.cat([base_ids, cont], dim=0))
            all_attn.append(torch.cat([base_attn, cont_attn], dim=0))

    all_input = torch.stack(all_input_ids, dim=0)
    all_mask = torch.stack(all_attn, dim=0)
    full_prompt_len = prompt_pad_len  # continuation starts right after this

    # Score in batches of (prompt, choice) pairs
    all_log_probs: List[float] = [0.0] * int(all_input.shape[0])
    for i in range(0, int(all_input.shape[0]), batch_size):
        batch_ids = all_input[i : i + batch_size].to(device)
        batch_mask = all_mask[i : i + batch_size].to(device)

        outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
        logits = outputs.logits  # (bsz, seq_len, vocab_size)

        bsz = int(batch_ids.shape[0])
        for b in range(bsz):
            clen = int(cont_lens[i + b])
            if clen <= 0:
                all_log_probs[i + b] = float("-inf")
                continue
            # First continuation token is at position `full_prompt_len`, so the
            # logits that predict it are at `full_prompt_len - 1`.
            pred_pos = torch.arange(
                full_prompt_len - 1,
                full_prompt_len - 1 + clen,
                device=device,
                dtype=torch.long,
            )
            tgt = batch_ids[b, full_prompt_len : full_prompt_len + clen]
            lp = torch.log_softmax(logits[b, pred_pos, :], dim=-1)
            lp_toks = lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
            all_log_probs[i + b] = float(lp_toks.sum().item())

    # Group log-probs back by example and take argmax
    predictions: List[int] = []
    for i in range(len(prompts)):
        lp = all_log_probs[i * num_choices : (i + 1) * num_choices]
        predictions.append(int(torch.tensor(lp).argmax().item()))

    return predictions

def evaluate_global(
    model,
    tokenizer,
    adapter_state: Dict[str, torch.Tensor],
    eval_examples: List[Dict[str, Any]],
    cfg: ExperimentConfig,
    max_examples: int = 200,
) -> Dict[str, Any]:
    """Evaluate the global adapter on the eval set.

    - arc_challenge / hellaswag: log-likelihood scoring (no generation).
      Predicted label = argmax log P(choice_letter | prompt).
      This matches lm-evaluation-harness / Open LLM Leaderboard methodology.

    - math_mcqa / all other datasets: generative evaluation.
      math_mcqa generates CoT reasoning then "Answer: X"; extract_mc_letter_final()
      pulls the final declared letter, matching the training-time oracle.
    """
    set_peft_model_state_dict(model, adapter_state, adapter_name="default")
    if hasattr(model, "set_adapter"):
        model.set_adapter("default")
    model.eval()

    ds = cfg.data.dataset
    results: Dict[str, Any] = {"dataset": ds, "n": 0}
    n = min(max_examples, len(eval_examples))
    correct = 0
    passed = 0
    f1_sum = 0.0

    # ------------------------------------------------------------------
    # MC tasks (arc_challenge, hellaswag): log-likelihood scoring
    # Predicted label = argmax log P(choice_letter | prompt).
    # Works because these models are trained to output a single letter
    # directly — the letter token is the expected first token of the output.
    # ------------------------------------------------------------------
    if ds in ("arc_challenge", "hellaswag"):
        bsz = max(1, int(getattr(cfg.gen, "eval_batch_size", 8) or 8))
        examples_to_eval = eval_examples[:n]

        for i in tqdm(range(0, n, bsz), desc="eval (log-likelihood)", leave=False):
            batch_ex = examples_to_eval[i : i + bsz]

            prompts = [ex["prompt"] for ex in batch_ex]

            pred_indices = _score_mc_logprobs_batched(
                model, tokenizer, prompts,
                num_choices=4,
                batch_size=bsz,
            )

            for ex, pred_idx in zip(batch_ex, pred_indices):
                gold_letter = (ex["meta"].get("reference") or "").strip().upper()
                pred_letter = MC_CHOICE_LABELS[pred_idx]
                correct += int(bool(gold_letter) and pred_letter == gold_letter)
                results["n"] += 1

        results["accuracy"] = correct / max(1, n)
        results["eval_method"] = "log_likelihood"
        return results

    # ------------------------------------------------------------------
    # All other tasks: generative evaluation
    # ------------------------------------------------------------------
    eval_do_sample = bool(getattr(cfg.gen, "eval_do_sample", False))
    params = GenParams(
        max_new_tokens=int(cfg.gen.max_new_tokens_y0),
        do_sample=eval_do_sample,
        temperature=float(getattr(cfg.gen, "eval_temperature",
                                  0.0 if not eval_do_sample else cfg.gen.temperature)),
        top_p=float(getattr(cfg.gen, "eval_top_p",
                            1.0 if not eval_do_sample else cfg.gen.top_p)),
        use_cache=True,
    )
    bsz = max(1, int(getattr(cfg.gen, "eval_batch_size", 1) or 1))

    grpo_strategyqa = (ds == "strategy_qa" and cfg.algorithm.name == "grpo")
    for i in tqdm(range(0, n, bsz), desc="eval (generative)", leave=False):
        batch_ex = eval_examples[i : i + bsz]
        if grpo_strategyqa:
            prompts = []
            for ex in batch_ex:
                p = ex["prompt"]
                facts = list((ex.get("meta") or {}).get("facts") or [])
                if facts:
                    facts_text = "\n".join(f"  - {f}" for f in facts)
                    p = p + f"[RELEVANT FACTS]\n{facts_text}\n\n"
                prompts.append(p)
        else:
            prompts = [ex["prompt"] for ex in batch_ex]
        outs = generate_batch_hf(model, tokenizer, prompts, params)

        for ex, out in zip(batch_ex, outs):
            if ds == "math_mcqa":
                # Model generates CoT reasoning then "Answer: X".
                # Use extract_mc_letter_final so intermediate letter mentions
                # in the chain of thought don't shadow the final declaration.
                pred = (extract_mc_letter_final(out) or "").upper()
                gold = (ex["meta"].get("reference") or "").strip().upper()
                correct += int(bool(gold) and pred == gold)

            elif ds == "strategy_qa":
                # Use extract_yes_no_final so intermediate yes/no mentions in
                # the reasoning chain don't shadow the final "Answer: yes/no"
                # declaration — mirrors how extract_mc_letter_final works for
                # math_mcqa.
                pred = extract_yes_no_final(out) or ""
                gold = (ex["meta"].get("answer") or
                        ex["meta"].get("reference") or "").strip().lower()
                correct += int(bool(gold) and pred == gold)

            results["n"] += 1

    if ds in ("math_mcqa", "strategy_qa"):
        results["accuracy"] = correct / max(1, n)
        results["eval_method"] = "generative"
    return results