from __future__ import annotations

"""Inference helpers.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch

from .prompting import format_prompt


@dataclass
class GenParams:
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    # Using KV cache is the standard fast path for decoding. Keep it explicit so callers
    # can override if they are extremely memory-constrained.
    use_cache: bool = True


def _decode_generated_only(tokenizer, output_ids: torch.Tensor, prompt_len: int) -> str:
    """Decode only the newly generated tokens.

    String-based prefix stripping is brittle because tokenization/decoding can
    normalize whitespace or drop characters, making `decoded.startswith(prompt)`
    unreliable. That can cause the *entire* prompt to be treated as model output,
    which breaks both evaluation (e.g., always extracting the 'A' from "A, B, C, D")
    and training rollouts (y0/y1 containing the prompt).
    """
    prompt_len = min(int(prompt_len), int(output_ids.shape[0]))
    gen_ids = output_ids[prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


@torch.inference_mode()
def generate_batch_hf(model, tokenizer, prompts: List[str], params: GenParams) -> List[str]:
    """Batch generation using a HF causal LM.

    Keeps memory usage modest by allowing the caller to choose batch size externally.
    """
    if not prompts:
        return []

    # Format prompts for chat models when applicable.
    formatted_prompts = [format_prompt(tokenizer, p) for p in prompts]

    # Tokenize as a padded batch.
    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Prompt lengths (number of non-pad tokens) so we can slice generated tokens
    # robustly instead of doing string prefix stripping.
    input_len = int(inputs["input_ids"].shape[1])  # padded prompt length
    # NOTE: tokenizers for causal LMs usually use left padding. In that case,
    # slicing by the *non-pad* length (attention_mask.sum) leaves part of the
    # prompt in the decoded output, breaking rewards/eval. We always slice at
    # the padded input length used by `generate`.


    out = model.generate(
        **inputs,
        max_new_tokens=int(params.max_new_tokens),
        do_sample=bool(params.do_sample),
        temperature=float(params.temperature),
        top_p=float(params.top_p),
        num_beams=1,
        use_cache=bool(params.use_cache),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode only the generated continuation for each prompt.
    return [_decode_generated_only(tokenizer, out[i], input_len) for i in range(out.shape[0])]
