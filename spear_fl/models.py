from __future__ import annotations

from typing import Any, Dict, Optional#, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import ModelConfig, GenerationConfig, HuggingFaceConfig
from .utils import dtype_from_str


def _hf_loader_kwargs(hf: Optional[HuggingFaceConfig]) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if hf is None:
        return kw
    if hf.model_cache_dir:
        kw["cache_dir"] = hf.model_cache_dir
    # If offline=True, do not hit the Hub (even for metadata).
    kw["local_files_only"] = bool(hf.offline)
    return kw


def load_tokenizer(cfg: ModelConfig, hf: Optional[HuggingFaceConfig] = None):
    name = cfg.tokenizer_name or cfg.model_name
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=cfg.trust_remote_code, **_hf_loader_kwargs(hf))
    # Padding setup for causal LM
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    # IMPORTANT: keep the *end* of the prompt (question + "Answer:" marker + chat
    # generation prompt) when truncating. With left padding, right-side truncation
    # can drop the answer region and silently break both training rollouts and eval.
    tok.truncation_side = "left"

    # Respect the experiment max sequence length instead of relying on the
    # tokenizer default (some tokenizers ship with a huge sentinel value).
    try:
        if int(cfg.max_seq_len) > 0:
            tok.model_max_length = int(cfg.max_seq_len)
    except Exception:
        pass
    return tok


def load_base_model(cfg: ModelConfig, tokenizer=None, hf: Optional[HuggingFaceConfig] = None):
    torch_dtype = dtype_from_str(cfg.dtype)

    kwargs = dict(
        trust_remote_code=cfg.trust_remote_code,
        device_map=cfg.device_map,
        torch_dtype=torch_dtype,
    )
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype)
        kwargs["quantization_config"] = bnb_cfg

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs, **_hf_loader_kwargs(hf))
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model


@torch.no_grad()
def generate(model, tokenizer, prompt: str, gen_cfg: GenerationConfig) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    kwargs = dict(
        max_new_tokens=gen_cfg.max_new_tokens_y0,
        do_sample=gen_cfg.do_sample,
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    out = model.generate(**inputs, **kwargs)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text


def strip_prompt_from_generation(full_text: str, prompt: str) -> str:
    # A simple heuristic: if full_text begins with prompt, strip it
    if full_text.startswith(prompt):
        return full_text[len(prompt):].lstrip()
    return full_text
