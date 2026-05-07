"""Dataset loading and example formatting.

Each loader returns (train_examples, eval_examples) where each example is a dict
with keys: id, prompt, meta (containing reference, reference_type, targets, etc.).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re

from datasets import load_dataset

from .config import DataConfig, HuggingFaceConfig
from .extractors import MC_LABELS


def _hf_dataset_kwargs(hf: Optional[HuggingFaceConfig]) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if hf is None:
        return kw
    if hf.dataset_cache_dir:
        kw["cache_dir"] = hf.dataset_cache_dir
    kw["download_mode"] = "reuse_cache_if_exists"
    return kw


# ---------------------------------------------------------------------------
# Per-dataset example constructors
# ---------------------------------------------------------------------------


def _arc_to_ex(row: Dict[str, Any], i: int) -> Dict[str, Any]:
    q = row["question"]
    choices = row["choices"]
    labels = list(choices.get("label") or [])
    texts = list(choices.get("text") or [])
    num_map = {"1": "A", "2": "B", "3": "C", "4": "D"}
    labels = [num_map.get(str(l).strip(), str(l).strip()) for l in labels]
    opt_lines = [f"{lbl}. {txt}" for lbl, txt in zip(labels, texts)]
    answer_raw = str(row.get("answerKey") or "").strip()
    answer = num_map.get(answer_raw, answer_raw)
    options_dict = {lbl: txt for lbl, txt in zip(labels, texts)}
    prompt = (
        "Answer the multiple-choice question by choosing the best option.\n"
        "Respond with ONLY the letter (A, B, C, or D).\n\n"
        + f"[QUESTION]\n{q}\n\n"
        + "[OPTIONS]\n" + "\n".join(opt_lines) + "\n\n"
        + "Answer:"
    )
    return {
        "id": f"arc_{i}",
        "prompt": prompt,
        "meta": {
            "answer": answer,
            "reference": answer or None,
            "reference_type": "mc_letter",
            "targets": answer or None,
            "options": opt_lines,
            "options_dict": options_dict,
            "question": q,
        },
    }


def _hellaswag_to_ex(row: Dict[str, Any], i: int) -> Dict[str, Any]:
    ctx = (row.get("ctx") or "").strip()
    endings = list(row.get("endings") or [])
    label_idx = int(row.get("label")) if row.get("label") is not None else None
    opt_lines = [f"{MC_LABELS[j]}. {endings[j]}" for j in range(min(4, len(endings)))]
    answer = MC_LABELS[label_idx] if isinstance(label_idx, int) and 0 <= label_idx < 4 else None
    options_dict = {MC_LABELS[j]: endings[j] for j in range(min(4, len(endings)))}
    prompt = (
        "Choose the best continuation of the context. Respond with ONLY the letter (A, B, C, or D).\n\n"
        + f"[CONTEXT]\n{ctx}\n\n"
        + "[OPTIONS]\n" + "\n".join(opt_lines) + "\n\n"
        + "Answer:"
    )
    return {
        "id": f"hellaswag_{i}",
        "prompt": prompt,
        "meta": {
            "label": label_idx,
            "reference": answer,
            "reference_type": "mc_letter",
            "targets": answer,
            "options": opt_lines,
            "options_dict": options_dict,
            "context": ctx,
        },
    }

# ---------------------------------------------------------------------------
# Math MCQA (stellaathena/math_mcqa)
# ---------------------------------------------------------------------------
def _math_mcqa_parse_choices(row: Dict[str, Any]) -> Tuple[List[str], Dict[str, str]]:
    """Parse answer choices from a math_mcqa row.

    Primary format (stellaathena/math_mcqa): 'choices' column is a plain Python
    list of 4 strings, e.g. ["12", "24", "36", "48"].  The 'answer' field is
    either a 0-based integer index, a 1-based integer, or a letter A-D.

    Also handles:
      - Individual columns named 'A', 'B', 'C', 'D'
      - ARC-style dict with 'label'/'text' sub-keys
    Returns (opt_lines, options_dict).
    """
    label_set = ["A", "B", "C", "D"]

    # Format 1 (primary for stellaathena/math_mcqa): 'choices' or 'options' as a list
    for key in ("choices", "options"):
        val = row.get(key)
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            items = list(val)
            if items:
                options_dict = {label_set[j]: str(items[j])
                                for j in range(min(4, len(items)))}
                opt_lines = [f"{lbl}. {options_dict[lbl]}" for lbl in label_set
                             if lbl in options_dict]
                return opt_lines, options_dict
        if isinstance(val, dict):
            # ARC-style: {"label": [...], "text": [...]}
            labels_raw = list(val.get("label") or [])
            texts_raw = list(val.get("text") or [])
            options_dict = {str(l).strip(): str(t) for l, t in zip(labels_raw, texts_raw)}
            opt_lines = [f"{l}. {t}" for l, t in options_dict.items()]
            return opt_lines, options_dict

    # Format 2: individual columns named 'A', 'B', 'C', 'D'
    if all(k in row for k in label_set):
        options_dict = {lbl: str(row[lbl]) for lbl in label_set if row.get(lbl) is not None}
        opt_lines = [f"{lbl}. {options_dict[lbl]}" for lbl in label_set if lbl in options_dict]
        return opt_lines, options_dict

    return [], {}


def _math_mcqa_parse_answer(answer_raw: str, options_dict: Dict[str, str]) -> str:
    """Normalise an answer field to a letter A-D.

    Handles:
      - Letters:  "A" / "a" / "B" / ...
      - 0-based index:  0, 1, 2, 3  (int or string)
      - 1-based index:  1, 2, 3, 4  (string only, legacy)
      - Answer text matching one of the choices (exact or stripped)
    """
    label_set = ["A", "B", "C", "D"]
    s = str(answer_raw).strip()

    # Direct letter
    if s.upper() in label_set:
        return s.upper()

    # Try integer (could be 0-based or 1-based)
    try:
        idx = int(s)
        if 0 <= idx <= 3:                 # 0-based
            return label_set[idx]
        if 1 <= idx <= 4:                 # 1-based
            return label_set[idx - 1]
    except (ValueError, TypeError):
        pass

    # Fallback: check if the answer text matches one of the choice values
    for lbl, text in options_dict.items():
        if s.lower() == str(text).strip().lower():
            return lbl

    # Last resort: return upper-cased as-is (may be wrong, but avoids silent failure)
    return s.upper()


def _math_mcqa_to_ex(row: Dict[str, Any], i: int) -> Dict[str, Any]:
    """Format a stellaathena/math_mcqa example.

    Expected columns:
      - problem / question: the math problem text
      - choices / options (list of 4 strings) — or individual A/B/C/D columns
      - answer: correct answer as 0-based index, letter, or choice text
      - solution: step-by-step solution (used for oracle feedback hints; \\boxed stripped)

    Prompt format: asks the model to reason step-by-step and then give
    "Answer: X" on the final line, where X is A/B/C/D.  Log-likelihood
    evaluation is unaffected (scores each letter directly after "Answer:").
    """
    problem = (
        row.get("problem") or row.get("question") or row.get("prompt")
        or row.get("input") or row.get("text") or ""
    )

    solution = str(row.get("solution") or row.get("explanation") or "")

    opt_lines, options_dict = _math_mcqa_parse_choices(row)

    answer_raw = str(row.get("answer") or row.get("correct_answer") or "").strip()
    answer = _math_mcqa_parse_answer(answer_raw, options_dict)

    # Training prompt: instructs the model to reason before committing to a letter.
    # The model generates a chain-of-thought, then ends with "Answer: X".
    prompt = (
        "Solve the math problem step by step, then choose the correct answer.\n"
        "Show your reasoning, then end with exactly: \"Answer: X\" "
        "where X is the letter A, B, C, or D.\n\n"
        + f"[PROBLEM]\n{problem}\n\n"
        + "[OPTIONS]\n" + "\n".join(opt_lines) + "\n\n"
        # + "Answer:"
    )

    ex_id = row.get("id") or row.get("task_id") or row.get("uid") or i
    return {
        "id": f"math_mcqa_{ex_id}",
        "prompt": prompt,
        "meta": {
            "answer": answer,
            "reference": answer or None,
            "reference_type": "mc_letter",
            "targets": answer or None,
            "options": opt_lines,
            "options_dict": options_dict,
            "problem": problem,
            "solution": solution,
        },
    }


# ---------------------------------------------------------------------------
# StrategyQA (ChilleD/StrategyQA)
# ---------------------------------------------------------------------------

def _strategyqa_to_ex(row: Dict[str, Any], i: int) -> Dict[str, Any]:
    """Format a ChilleD/StrategyQA example.

    Expected columns:
      - question: the yes/no question text
      - answer: bool (True = yes, False = no)
      - facts: list of supporting facts (used as oracle feedback hints)
      - decomposition: list of sub-questions (optional)

    Prompt asks the model to reason briefly and answer with exactly
    "yes" or "no" on the final line.
    """
    question = str(row.get("question") or "")
    answer_raw = row.get("answer")
    # answer is a bool in the HF dataset
    if isinstance(answer_raw, bool):
        answer_str = "yes" if answer_raw else "no"
    elif isinstance(answer_raw, str):
        answer_str = "yes" if answer_raw.strip().lower() in ("true", "yes", "1") else "no"
    else:
        answer_str = "yes" if bool(answer_raw) else "no"

    facts = list(row.get("facts") or [])

    prompt = (
        "Answer the following yes/no question by reasoning step by step.\n"
        "Think through the question carefully, then end your response with "
        "exactly: \"Answer: yes\" or \"Answer: no\" on the final line.\n\n"
        + f"[QUESTION]\n{question}\n\n"
    )

    qid = row.get("qid") or row.get("id") or i
    return {
        "id": f"strategy_qa_{qid}",
        "prompt": prompt,
        "meta": {
            "question": question,
            "answer": answer_str,
            "facts": facts,
            "reference": answer_str,
            "reference_type": "yes_no",
            "targets": answer_str,
        },
    }


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------
def load_benchmark(
    cfg: DataConfig, hf: Optional[HuggingFaceConfig] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load (train, eval) example lists for the configured dataset."""
    kw = _hf_dataset_kwargs(hf)
    seed = getattr(cfg, "seed", 42)

    if cfg.dataset == "arc_challenge":
        ds_train = load_dataset("ai2_arc", "ARC-Challenge", split=cfg.split, **kw).shuffle(seed=seed)
        ds_val = load_dataset("ai2_arc", "ARC-Challenge", split="test", **kw).shuffle(seed=seed)
        return (
            [_arc_to_ex(ds_train[i], i) for i in range(len(ds_train))],
            [_arc_to_ex(ds_val[i], i) for i in range(len(ds_val))],
        )

    if cfg.dataset == "hellaswag":
        ds_train = load_dataset("hellaswag", split=cfg.split, **kw).shuffle(seed=seed)
        ds_val = load_dataset("hellaswag", split="validation", **kw).shuffle(seed=seed)
        return (
            [_hellaswag_to_ex(ds_train[i], i) for i in range(len(ds_train))],
            [_hellaswag_to_ex(ds_val[i], i) for i in range(len(ds_val))],
        )

    if cfg.dataset == "math_mcqa":
        ds_all = load_dataset("stellaathena/math_mcqa", **kw)

        # Try common split names for train
        ds_train = None
        for sp in [cfg.split, "train"]:
            if sp in ds_all:
                try:
                    ds_train = ds_all[sp].shuffle(seed=seed)
                    break
                except Exception:
                    pass
        if ds_train is None:
            first_key = list(ds_all.keys())[0]
            ds_train = ds_all[first_key].shuffle(seed=seed)

        # Try to get a separate eval split
        eval_split = cfg.eval_split or "test"
        ds_eval = None
        train_split_used = cfg.split if cfg.split in ds_all else list(ds_all.keys())[0]
        for sp in [eval_split, "validation", "val", "test", "dev"]:
            if sp in ds_all and sp != train_split_used:
                try:
                    ds_eval = ds_all[sp].shuffle(seed=seed)
                    break
                except Exception:
                    pass

        train = [_math_mcqa_to_ex(ds_train[i], i) for i in range(len(ds_train))]

        if ds_eval is not None:
            eval_ = [_math_mcqa_to_ex(ds_eval[i], i) for i in range(len(ds_eval))]
        else:
            n_eval = max(1, int(len(train) * float(cfg.eval_fraction)))
            eval_ = train[-n_eval:]
            train = train[:-n_eval]

        return train, eval_

    if cfg.dataset == "strategy_qa":
        ds_all = load_dataset("ChilleD/StrategyQA", **kw)

        ds_train = None
        for sp in [cfg.split, "train"]:
            if sp in ds_all:
                try:
                    ds_train = ds_all[sp].shuffle(seed=seed)
                    break
                except Exception:
                    pass
        if ds_train is None:
            first_key = list(ds_all.keys())[0]
            ds_train = ds_all[first_key].shuffle(seed=seed)

        eval_split = cfg.eval_split or "validation"
        ds_eval = None
        train_split_used = cfg.split if cfg.split in ds_all else list(ds_all.keys())[0]
        for sp in [eval_split, "validation", "val", "test", "dev"]:
            if sp in ds_all and sp != train_split_used:
                try:
                    ds_eval = ds_all[sp].shuffle(seed=seed)
                    break
                except Exception:
                    pass

        train = [_strategyqa_to_ex(ds_train[i], i) for i in range(len(ds_train))]

        if ds_eval is not None:
            eval_ = [_strategyqa_to_ex(ds_eval[i], i) for i in range(len(ds_eval))]
        else:
            n_eval = max(1, int(len(train) * float(cfg.eval_fraction)))
            eval_ = train[-n_eval:]
            train = train[:-n_eval]

        return train, eval_

    raise ValueError(f"Unknown dataset: {cfg.dataset}")


def build_partition_labels(examples: List[Dict[str, Any]], cfg: DataConfig) -> List[int]:
    """Build categorical labels for Dirichlet partitioning."""
    nc = int(cfg.dirichlet_num_classes)
    labels: List[int] = []
    for ex in examples:
        ds = cfg.dataset
        if ds in ("arc_challenge", "hellaswag", "math_mcqa"):
            a = ex["meta"].get("answer") or ex["meta"].get("reference") or "A"
            labels.append((ord(str(a)[0].upper()) - ord("A")) % max(1, nc))
        elif ds == "strategy_qa":
            # binary label: yes=1, no=0
            a = ex["meta"].get("answer") or "no"
            labels.append(1 if str(a).lower() == "yes" else 0)
        else:
            labels.append(abs(hash(ex["id"])) % nc)
    return labels