"""Feedback providers: oracle

Oracle feedback guides the model WITHOUT revealing the final answer.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from .config import FeedbackConfig, DataConfig, HuggingFaceConfig
from .extractors import (
    extract_mc_letter,
    extract_yes_no_final,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hf_loader_kwargs(hf: Optional[HuggingFaceConfig]) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if hf is None:
        return kw
    if hf.model_cache_dir:
        kw["cache_dir"] = hf.model_cache_dir
    kw["local_files_only"] = bool(hf.offline)
    return kw

def _first_n_words(text: str, n: int) -> str:
    """Return the first *n* words of *text*."""
    words = text.split()
    return " ".join(words[:n])


def _remove_boxed_from_solution(text: str) -> str:
    """Remove all \\boxed{...} occurrences from a solution string (handles nested braces)."""
    import re as _re
    result = []
    i = 0
    s = text
    while i < len(s):
        m = _re.search(r"\\boxed\s*\{", s[i:])
        if not m:
            result.append(s[i:])
            break
        result.append(s[i: i + m.start()])
        start = i + m.end()
        depth = 1
        j = start
        while j < len(s) and depth > 0:
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
            j += 1
        i = j
    return "".join(result)


def _extract_solution_hint(text: str, gold_answer: str, max_lines: int = 6,
                           max_chars: int = 600) -> str:
    """Extract the first few non-trivial lines of a solution as a hint,
    filtering out any line that contains the gold answer string."""
    if not text:
        return ""
    lines = text.split("\n")
    hint_lines: List[str] = []
    char_count = 0
    for line in lines:
        clean = line.strip()
        if not clean or char_count >= max_chars:
            break
        if len(hint_lines) >= max_lines:
            break
        # Skip lines that appear to directly state the answer
        if gold_answer and str(gold_answer).strip():
            if str(gold_answer).strip().lower() in clean.lower():
                continue
        hint_lines.append(f"  {clean}")
        char_count += len(clean)
    return "\n".join(hint_lines) if hint_lines else ""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class FeedbackProviderBase:
    def feedback(self, ex: Dict[str, Any], y0: str, data_cfg: DataConfig,
                 round_idx: int) -> str:
        raise NotImplementedError

    def feedback_batch(self, exs: List[Dict[str, Any]], y0s: List[str],
                       data_cfg: DataConfig, round_idx: int) -> List[str]:
        return [self.feedback(ex, y0, data_cfg, round_idx)
                for ex, y0 in zip(exs, y0s)]


# ---------------------------------------------------------------------------
# Oracle feedback
# ---------------------------------------------------------------------------

class OracleFeedback(FeedbackProviderBase):
    """Oracle feedback using ground truth WITHOUT revealing final answers."""

    def __init__(self, mc_hint_words: int = 3):
        self.mc_hint_words = mc_hint_words

    def feedback(self, ex: Dict[str, Any], y0: str, data_cfg: DataConfig,
                 round_idx: int) -> str:
        ds = data_cfg.dataset
        meta = ex.get("meta", {})

        if ds == "math_mcqa":
            return self._math_mcqa(y0, meta)
        if ds in ("arc_challenge", "hellaswag"):
            return self._mc(y0, meta, ds)
        if ds == "strategy_qa":
            return self._strategy_qa(y0, meta)
        return "Please revise your answer."

    # ── Math MCQA ─────────────────────────────────────────────────────
    def _math_mcqa(self, y0: str, meta: dict) -> str:
        """Feedback for math_mcqa.

        Hints come from the first few lines of the gold solution column,
        with all \\boxed{...} stripped so the answer value is not leaked.
        Feedback always reminds the model to reason and end with "Answer: X".
        """
        pred = (extract_mc_letter(y0) or "").upper()
        gold = (meta.get("reference") or meta.get("answer") or "").strip().upper()
        options_dict = meta.get("options_dict", {})
        solution = meta.get("solution", "") or ""

        if pred == gold and gold:
            return (f"Correct! The answer is {gold}. "
                    "Remember to end with \"Answer: X\" on the final line.")

        # Strip \boxed{{...}} so the answer value is not leaked, then take
        # the first few non-trivial lines as a reasoning scaffold hint.
        solution_no_boxed = _remove_boxed_from_solution(solution)
        hint_lines = _extract_solution_hint(solution_no_boxed, gold, max_lines=3, max_chars=300)

        correct_text = options_dict.get(gold, "")
        hint_words = _first_n_words(correct_text, self.mc_hint_words)

        fb = f"Incorrect. Your answer \'{pred}\' is wrong."
        if hint_lines:
            fb += f"\n\nHint — consider these opening steps of the solution:\n{hint_lines}"
        elif hint_words:
            fb += f"\n\nHint: Consider an answer starting with \"{hint_words}...\""
        else:
            fb += "\n\nHint: re-read the problem and each option carefully."

        fb += (
            "\n\nReason step by step, then end your response with exactly: "
            "\"Answer: X\" where X is A, B, C, or D. "
            "DO NOT repeat the letter you used previously."
        )
        return fb

    # ── MC (ARC / HellaSwag) ──────────────────────────────────────────
    def _mc(self, y0: str, meta: dict, ds: str) -> str:
        pred = (extract_mc_letter(y0) or "").upper()
        gold = (meta.get("reference") or meta.get("answer") or "").strip().upper()
        options_dict = meta.get("options_dict", {})

        if pred == gold and gold:
            return f"Correct! The answer is {gold}."

        correct_text = options_dict.get(gold, "")
        hint_words = _first_n_words(correct_text, self.mc_hint_words)

        fb = f"Incorrect. '{pred}' is wrong."
        if hint_words:
            fb += (f"\n\nHint: Consider an answer that starts with \"{hint_words}...\"")
        else:
            fb += "\n\nHint: re-read the question and options carefully."

        if ds == "arc_challenge":
            fb += " Consider the underlying scientific principle."
        elif ds == "hellaswag":
            fb += " Choose the most natural continuation."

        fb += "\n\nAnswer with exactly one letter: A, B, C, or D. DO NOT choose the answer you used previously."
        return fb

    # ── StrategyQA ────────────────────────────────────────────────────
    def _strategy_qa(self, y0: str, meta: dict) -> str:
        """Feedback for StrategyQA (yes/no reasoning questions).

        Uses extract_yes_no_final so that yes/no mentions inside the reasoning
        chain are ignored — only the final 'Answer: yes/no' declaration counts.
        Hints come from a partial subset of the 'facts' column (similar to how
        math_mcqa uses opening lines of the solution).
        """
        pred = extract_yes_no_final(y0) or ""
        gold = (meta.get("answer") or meta.get("reference") or "").strip().lower()
        facts: List[str] = list(meta.get("facts") or [])

        if pred == gold and gold:
            return (
                f"Correct! The answer is '{gold}'. "
                "Remember to end with exactly: \"Answer: yes\" or \"Answer: no\"."
            )

        # Make feedback the facts. Show half of it, user doesnt have full picture
        num_facts_to_show = max(1, len(facts) // 2) if facts else 0
        hint_facts = facts[:num_facts_to_show]

        fb = f"Incorrect. '{pred}' is not the right answer."
        if hint_facts:
            facts_text = "\n".join(f"  - {f}" for f in hint_facts)
            fb += f"\n\nHint — consider these relevant facts:\n{facts_text}"
        else:
            fb += "\n\nHint: Think carefully about the reasoning required to answer this question."

        fb += (
            "\n\nReason through the question step by step, then end your response "
            "with exactly: \"Answer: yes\" or \"Answer: no\". "
            "DO NOT repeat the answer you gave previously."
        )
        return fb


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_feedback_provider(fb_cfg: FeedbackConfig,
                            hf: Optional[HuggingFaceConfig] = None
                            ) -> FeedbackProviderBase:
    mc_words = getattr(fb_cfg, "mc_hint_words", 3)
    if fb_cfg.provider == "oracle":
        base: FeedbackProviderBase = OracleFeedback(mc_hint_words=mc_words)
    else:
        raise ValueError(f"Unknown feedback provider: {fb_cfg.provider}")
    return base