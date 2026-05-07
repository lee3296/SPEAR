from __future__ import annotations

import re

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# ---------------------------------------------------------------------------
# Model-type detection
# ---------------------------------------------------------------------------

# Patterns that reliably identify instruction-tuned / chat models.
# Base models from Llama 3+ and Qwen 2.5+ ship with a chat_template in their
# tokenizer config even though they were NEVER trained on that format.
# Blindly applying apply_chat_template() on a base model wraps every prompt in
# special role tokens the model assigns near-random probabilities to, which
# breaks both generation quality and log-likelihood scoring.
_INSTRUCT_PATTERNS = re.compile(
    r"instruct|chat|it\b|-sft\b|tulu|assistant|rlhf|dpo|orca|alpaca|vicuna|wizard",
    re.IGNORECASE,
)


def is_chat_model(tokenizer) -> bool:
    """Return True only if this model was actually trained on a chat format.

    Checks the tokenizer's name_or_path (i.e. the HuggingFace model ID or local
    path) for known instruct/chat suffixes.  Falls back to False (treat as base)
    when the name is unavailable, which is the safe default — applying a chat
    template to a base model actively harms quality, while skipping it on a chat
    model merely means the prompt is slightly less idiomatic (still works fine
    because the model has seen plain-text during pretraining too).
    """
    name = getattr(tokenizer, "name_or_path", "") or ""
    # For local paths, use only the final component (e.g. /models/Qwen2.5-1.5B-Instruct)
    name = name.rstrip("/").split("/")[-1]
    return bool(_INSTRUCT_PATTERNS.search(name))


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(
    tokenizer,
    user_prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Format a prompt for either a chat/instruct model or a base causal LM.

    For genuine instruct models (detected via model name):
        Wraps the prompt in a system + user message using apply_chat_template,
        ending with the assistant generation prompt so decoding starts cleanly.

    For base models (including Llama-3.2 base, Qwen2.5 base, etc.):
        Returns the prompt unchanged.  These models ship with a chat_template
        in their tokenizer but were NOT trained on it — applying it corrupts
        generation and makes log-likelihood scores near-random.

    Note: evaluation can use either generation or log-likelihood scoring.
    For log-likelihood scoring we also apply this function so the choice token
    is scored in the same chat-template context the model was trained on.
    """
    if user_prompt is None:
        return ""
    user_prompt = str(user_prompt)

    # Only apply the chat template when the model was actually trained on it.
    if not is_chat_model(tokenizer):
        return user_prompt

    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None or not getattr(tokenizer, "chat_template", None):
        return user_prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return apply(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    except ValueError:
        return user_prompt
    except TypeError:
        try:
            rendered = apply(messages, tokenize=False)
        except TypeError:
            rendered = apply(messages)
        return rendered
