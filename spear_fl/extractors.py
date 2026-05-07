"""Shared answer extraction and normalization utilities.

Consolidates duplicated helpers from feedback.py, metrics.py, algorithms.py, and focus.py
into a single canonical module. All other modules should import from here.
"""
from __future__ import annotations

from typing import List, Optional
import re


# ---------------------------------------------------------------------------
# Label / classification extraction
# ---------------------------------------------------------------------------
MC_LABELS = ["A", "B", "C", "D"]


def extract_mc_letter(text: str) -> Optional[str]:
    """Extract A/B/C/D from model output (first match)."""
    if not text:
        return None
    s = str(text).strip().upper()
    if not s:
        return None
    for pat in [
        r"^\s*([ABCD])(?:\b|[\)\.\:\-])",
        r"(?:ANSWER|OPTION)\s*[:\-]?\s*\(?\s*([ABCD])\s*\)?",
        r"\(\s*([ABCD])\s*\)",
        r"\b([ABCD])\b",
    ]:
        m = re.search(pat, s)
        if m:
            return m.group(1)
    return None


def extract_mc_letter_final(text: str) -> Optional[str]:
    """Extract the *final* A/B/C/D answer from a reasoning chain.

    Reasoning chains often mention option letters during intermediate steps
    (e.g. "Option A would be... but B is incorrect...") before the definitive
    answer declaration.  This function prioritises the last explicit
    "Answer: X" marker, then falls back to the last bare letter found.
    This prevents intermediate mentions from shadowing the real answer.
    """
    if not text:
        return None
    s = str(text).strip().upper()
    if not s:
        return None

    # Priority 1: last explicit "Answer: X" / "Answer - X" declaration
    hits = list(re.finditer(
        r"(?:ANSWER|OPTION)\s*[:\-]\s*\(?\s*([ABCD])\s*\)?", s
    ))
    if hits:
        return hits[-1].group(1)

    # Priority 2: last bare letter (handles models that just end with the letter)
    hits = list(re.finditer(r"\b([ABCD])\b", s))
    if hits:
        return hits[-1].group(1)

    return None


def extract_yes_no_final(text: str) -> Optional[str]:
    """Extract the *final* yes/no answer from a reasoning chain.

    Mirrors extract_mc_letter_final for StrategyQA-style tasks.  The model is
    instructed to reason step-by-step and end with 'Answer: yes' or
    'Answer: no', so intermediate mentions of yes/no during reasoning are
    ignored in favour of the last explicit declaration.

    Priority order:
      1. Last explicit 'Answer: yes/no' marker.
      2. Last bare 'yes' or 'no' word (fallback for models that skip the marker).
    """
    if not text:
        return None
    s = str(text).strip().lower()
    if not s:
        return None

    # Priority 1: last explicit "Answer: yes" / "Answer: no" declaration
    hits = list(re.finditer(
        r"answer\s*[:\-]\s*(yes|no)\b", s
    ))
    if hits:
        return hits[-1].group(1)

    # Priority 2: last bare yes/no word
    hits = list(re.finditer(r"\b(yes|no)\b", s))
    if hits:
        return hits[-1].group(1)

    return None


# ---------------------------------------------------------------------------
# Format validation
# ---------------------------------------------------------------------------

def has_valid_answer(y: str, dataset: str) -> bool:
    """Check if output has a valid extractable answer for the dataset."""
    if not y or len(y.strip()) < 2:
        return False
    y = y.strip()

    if dataset in ("arc_challenge", "hellaswag", "math_mcqa"):
        return bool(re.search(r"\b[ABCD]\b", y.upper()))
    if dataset == "strategy_qa":
        return bool(re.search(r"\b(yes|no)\b", y.lower()))
    return True