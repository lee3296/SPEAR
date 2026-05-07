"""Configuration dataclasses and YAML loader."""
from __future__ import annotations
 
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
 
import yaml
 
 
DatasetName = Literal[
    "arc_challenge", "hellaswag",
    "math_mcqa", "strategy_qa",
]
FeedbackProvider = Literal["oracle"]
AlgorithmName = Literal[
    "spear",            # ours
    "grpo",             # baseline: group RL
    # Baselines from other_methods.zip (papers):
    "opsd",             # On-Policy Self-Distillation
    "rltf_sd",          # RL from Text Feedback (Self-Distillation)
]
 
 
@dataclass
class AlgorithmConfig:
    name: AlgorithmName = "spear"
 
@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B"
    tokenizer_name: Optional[str] = None
    trust_remote_code: bool = True
    max_seq_len: int = 2048
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    device_map: str = "auto"
    load_in_4bit: bool = False
    gradient_checkpointing: bool = True
 
 
@dataclass
class LoraCfg:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    last_n_layers: int = 6
    target_modules: List[str] = field(default_factory=list)
 
 
@dataclass
class DataConfig:
    dataset: DatasetName = "arc_challenge"
    split: str = "train"
    eval_split: Optional[str] = None
    eval_fraction: float = 0.02
    max_attempts: int = 3 #no longer used, use num_revisions
    num_clients: int = 20
    dirichlet_alpha: float = 0.3
    dirichlet_num_classes: int = 10
    max_examples_per_client: Optional[int] = 500
    seed: int = 42
    eval_every: int = 5
    eval_every_round_up_to: int = 10
 
 
@dataclass
class FLConfig:
    rounds: int = 20
    clients_per_round: int = 5
    local_train_steps: int = 50
    local_rollout_steps: int = 25
    aggregation: Literal["fedavg", "fedadam", "fedyogi"] = "fedavg"
    fedprox_mu: float = 0.0
 
 
@dataclass
class GenerationConfig:
    max_new_tokens_y0: int = 256
    max_new_tokens_y1: int = 256
    num_revisions: int = 1  # number of feedback-guided rewrite attempts (y1, y2, ...)
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.95
    rollout_batch_size: int = 8
    eval_batch_size: int = 8
 
 
@dataclass
class SPEARConfig:
    """SPEAR-specific hyperparameters."""
    lambda_win: float = 1.0         # weight for SFT on correct completions
    lambda_lose: float = 0.3        # weight for unlikelihood on wrong completions
    lambda_anchor: float = 0.01     # weight for L2 anchor to reference
    unlikelihood_margin: float = 0.1  # only penalise tokens with p > margin
    ema_decay: float = 0.0          # EMA smoothing across rounds (0 = off)
    # If True, apply the unlikelihood (UL) loss only on the *answer suffix*
    # of an incorrect completion (e.g. the final
    # "\\boxed{...}" expression), instead of the whole chain-of-thought.
    # This prevents UL from suppressing shared reasoning tokens that appear in
    # both correct and incorrect solutions, which can otherwise cause
    # monotonic capability collapse on generative reasoning tasks.
    #
    # When enabled, UL uses an auto-detected answer suffix; if it cannot be
    # detected for a sample, it falls back to `ul_tail_tokens`.
    ul_answer_only: bool = False
    ul_tail_tokens: int = 0         # if > 0, apply UL loss only to the last N
                                    # completion tokens (the answer region).
                                    # Set to ~32 for generative tasks 
                                    # (math) to avoid suppressing shared CoT
                                    # tokens. 0 = apply to full completion.
    ul_skip_math_ops: bool = False  # if True, exclude mathematical operators
                                    # (+, -, *, /, =, (, ), ^, etc.) and LaTeX
                                    # structural tokens from UL for gsm8k and
                                    # math_lighteval.  These tokens appear in
                                    # ALL completions (correct and incorrect), so
                                    # suppressing them damages arithmetic
                                    # capability; the actual wrong content is the
                                    # specific wrong numeric value, not the
                                    # operators connecting it.
    max_lose_ratio: float = 2.0     # cap: lose_traces are subsampled so that
                                    # len(lose) <= max_lose_ratio * len(win).
                                    # Prevents UL from dominating SFT when the
                                    # model is struggling (many wrong answers).
                                    # Training is skipped entirely if wins == 0
                                    # since pure UL with nothing to reinforce
                                    # only degrades the model.
 
 
@dataclass
class GRPOConfig:
    num_generations: int = 4
    beta_kl: float = 0.02
    clip_range: float = 0.2
    normalize_advantage: bool = True
    reward_scale: float = 1.0
    reward_clip: Optional[float] = None
 
 
@dataclass
class RLTFSDConfig:
    """Hyperparameters for RLTF-SD (Algorithm 1, Song et al. 2026).
 
    Self Distillation: trains π(·|x0) to imitate its own feedback-conditioned
    second-turn generations y1, using a first-turn mean baseline to avoid
    gradient-signal collapse when the second-turn success rate is high.
 
    Paper reference: "Expanding the Capabilities of Reinforcement Learning
    via Text Feedback", Song et al. (arXiv:2602.02482v2, Feb 2026).
    """
    # Discount factor for computing returns R_i = r0_i + gamma * r1_i.
    # Paper (Alg. 1, line 10) uses this but does not specify a value; default
    # 1.0 gives undiscounted sum (equal weight to both turns).
    gamma: float = 1.0
 
    # Weight on the self-distillation gradient g_SD (Algorithm 1, line 15).
    # Paper Table 2 lists "RL coefficient (Self Distillation) 0.1", which we
    # interpret as the weight applied to the SD auxiliary loss relative to the
    # main multi-turn RL objective.
    sd_coeff: float = 0.1
 
    # Weight on the multi-turn RL gradient g_RL (Algorithm 1, line 16).
    # Scaling both SD and RL allows independent tuning. Set rl_coeff=0 to
    # run pure SD distillation without the RL objective.
    rl_coeff: float = 1.0
 
    # When True, episodes where r0=1 (y0 already correct) are terminated
    # early: no feedback is requested, and the second turn is skipped.
    # The paper (Section 2) mentions early termination as a realistic option.
    early_termination: bool = True
 
 
@dataclass
class DistillConfig:
    """Shared hyperparameters for self-distillation baselines."""
    temperature: float = 2.0
    kl_weight: float = 1.0
    anchor_weight: float = 0.0      # optional L2 anchor to global adapter
    priv_max_chars: int = 2000      # truncate privileged text / demonstrations
    feedback_max_chars: int = 800   # truncate feedback text in RLTF
    num_demos: int = 1              
    demo_max_chars: int = 1800
 
 
@dataclass
class OptimConfig:
    lr: float = 2e-4
    weight_decay: float = 1e-3
    batch_size: int = 1
    grad_accum_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    lr_min: float = 1e-5
    use_warmup: bool = False
    micro_batch_size: Optional[int] = None
 
 
@dataclass
class FeedbackConfig:
    provider: FeedbackProvider = "oracle"
    mc_hint_words: int = 3  # number of leading words for MC option hints
 
 
@dataclass
class HuggingFaceConfig:
    offline: bool = False
    hf_home: Optional[str] = None
    model_cache_dir: Optional[str] = None
    dataset_cache_dir: Optional[str] = None
    hub_cache_dir: Optional[str] = None
 
@dataclass
class ExperimentConfig:
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraCfg = field(default_factory=LoraCfg)
    data: DataConfig = field(default_factory=DataConfig)
    fl: FLConfig = field(default_factory=FLConfig)
    gen: GenerationConfig = field(default_factory=GenerationConfig)
    spear: SPEARConfig = field(default_factory=SPEARConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    rltf_sd: RLTFSDConfig = field(default_factory=RLTFSDConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    hf: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    output_dir: str = "runs/exp"
 
# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
 
def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out
 
 
def _dc_to_dict(dc) -> Any:
    if hasattr(dc, "__dataclass_fields__"):
        return {k: _dc_to_dict(getattr(dc, k)) for k in dc.__dataclass_fields__}
    if isinstance(dc, list):
        return [_dc_to_dict(x) for x in dc]
    return dc
 
 
def config_from_dict(d: Dict[str, Any]) -> ExperimentConfig:
    # Filter each sub-dict to only include known fields for that dataclass
    def _filter(dc_cls, sub: dict) -> dict:
        known = {f.name for f in dc_cls.__dataclass_fields__.values()}
        return {k: v for k, v in sub.items() if k in known}
 
    return ExperimentConfig(
        algorithm=AlgorithmConfig(**_filter(AlgorithmConfig, d.get("algorithm", {}))),
        model=ModelConfig(**_filter(ModelConfig, d.get("model", {}))),
        lora=LoraCfg(**_filter(LoraCfg, d.get("lora", {}))),
        data=DataConfig(**_filter(DataConfig, d.get("data", {}))),
        fl=FLConfig(**_filter(FLConfig, d.get("fl", {}))),
        gen=GenerationConfig(**_filter(GenerationConfig, d.get("gen", {}))),
        spear=SPEARConfig(**_filter(SPEARConfig, d.get("spear", {}))),
        grpo=GRPOConfig(**_filter(GRPOConfig, d.get("grpo", {}))),
        rltf_sd=RLTFSDConfig(**_filter(RLTFSDConfig, d.get("rltf_sd", {}))),
        distill=DistillConfig(**_filter(DistillConfig, d.get("distill", {}))),
        optim=OptimConfig(**_filter(OptimConfig, d.get("optim", {}))),
        feedback=FeedbackConfig(**_filter(FeedbackConfig, d.get("feedback", {}))),
        hf=HuggingFaceConfig(**_filter(HuggingFaceConfig, d.get("hf", {}))),
        output_dir=str(d.get("output_dir", "runs/exp")),
    )
 
def load_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    base = _dc_to_dict(ExperimentConfig())
    merged = _deep_update(base, user)
    return config_from_dict(merged)