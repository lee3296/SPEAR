# SPEAR: Self-Play Enhancement via Advantage-weighted Refinement in Online Federated LLM Fine-Tuning with Real-Time Feedback

## Overview

This repository contains the implementation of **SPEAR**, a federated learning framework for improving large language model (LLM) reasoning through feedback-guided self-play.

SPEAR is an efficient online learning algorithm that uses feedback-guided contrastive training pairs. It pushes probability toward correct completions and away from confident wrong answers, without requiring privileged teacher context.

**Supported algorithms:**
- `spear` — our method
- `grpo` — Group Relative Policy Optimisation (baseline)
- `opsd` — On-Policy Self-Distillation (baseline)
- `rltf_sd` — RL from Text Feedback via Self-Distillation (baseline)
- `feedback_sft` — SFT with feedback-augmented prompts (baseline)

**Supported datasets:** `arc_challenge`, `hellaswag`, `math_mcqa`, `strategy_qa`

**Supported models:** configs provided for Qwen2.5-1.5B and LLaMA-3.2-3B

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run an experiment

```bash
python -m spear_fl.train --config configs/math_mcqa_qwen1.5b_spear.yaml
```

The `--config` flag points to a YAML configuration file. Pre-built configs are provided in `configs/` for all dataset/model/algorithm combinations.

---

## Configuration

All experiment settings are controlled via a YAML file. Below is a full annotated reference:

```yaml
output_dir: runs/my_experiment   # where results and checkpoints are saved

algorithm:
  name: spear                    # spear | grpo | opsd | rltf_sd | feedback_sft

model:
  model_name: Qwen/Qwen2.5-1.5B # HuggingFace model ID
  dtype: bf16                    # bf16 | fp16 | fp32
  max_seq_len: 2048
  load_in_4bit: false
  gradient_checkpointing: true

lora:
  r: 16                          # LoRA rank
  alpha: 32                      # LoRA alpha
  dropout: 0.1
  last_n_layers: 36              # apply LoRA to the last N transformer layers

data:
  dataset: math_mcqa             # arc_challenge | hellaswag | math_mcqa | strategy_qa
  num_clients: 50                # number of federated clients
  dirichlet_alpha: 1.0           # heterogeneity of data partition (lower = more heterogeneous)
  max_examples_per_client: 500
  seed: 42
  eval_every: 5                  # evaluate global model every N rounds
  eval_every_round_up_to: 1      # always evaluate for the first N rounds

fl:
  rounds: 50                     # total federated rounds
  clients_per_round: 3           # clients sampled per round
  local_rollout_steps: 80        # rollout steps per client per round
  local_train_steps: 5           # gradient update steps per client per round
  aggregation: fedavg            # fedavg | fedadam | fedyogi

gen:
  max_new_tokens_y0: 512         # max tokens for initial generation
  max_new_tokens_y1: 512         # max tokens for revised generation
  num_revisions: 2               # number of feedback-guided revision attempts
  do_sample: true
  temperature: 0.8
  rollout_batch_size: 32
  eval_batch_size: 32

spear:                           # SPEAR-specific hyperparameters
  lambda_win: 1.0                # weight for MLE loss on correct completions
  lambda_lose: 0.1               # weight for unlikelihood loss on wrong completions
  lambda_anchor: 0.0             # L2 anchor weight to reference model
  unlikelihood_margin: 0.3       # only penalise tokens with p > margin
  ul_answer_only: false          # apply unlikelihood only to the answer suffix
  ul_skip_math_ops: true         # protect math operators in chain-of-thought
  ul_tail_tokens: 32
  max_lose_ratio: 4.0

optim:
  lr: 5.0e-5
  batch_size: 4
  grad_accum_steps: 4
  warmup_ratio: 0.1
  lr_min: 1.0e-5
  weight_decay: 0.0
  use_warmup: true

feedback:
  provider: oracle               
  mc_hint_words: 10              # number of hint words from correct MC option, fallback for MathMCQA

hf:                              # HuggingFace cache / offline settings
  offline: false
  hf_home: null
  model_cache_dir: null
  dataset_cache_dir: null
  hub_cache_dir: null
```

### CLI overrides for HuggingFace settings

```bash
python -m spear_fl.train \
  --config configs/math_mcqa_qwen1.5b_spear.yaml \
  --hf_home /path/to/cache \
  --hf_offline               # use only locally cached models/datasets
```

---

## Pre-built configs

| Dataset       | Model            | Algorithm    | Config path |
|---------------|------------------|--------------|-------------|
| ARC-Challenge | Qwen2.5-1.5B     | SPEAR        | `configs/arc_challenge_qwen1.5b_spear.yaml` |
| ARC-Challenge | LLaMA-3.2-3B     | SPEAR        | `configs/arc_challenge_llama3b_spear.yaml` |
| HellaSwag     | Qwen2.5-1.5B     | SPEAR        | `configs/hellaswag_qwen1.5b_spear.yaml` |
| HellaSwag     | LLaMA-3.2-3B     | SPEAR        | `configs/hellaswag_llama3b_spear.yaml` |
| MATH-MCQA     | Qwen2.5-1.5B     | SPEAR        | `configs/math_mcqa_qwen1.5b_spear.yaml` |
| MATH-MCQA     | LLaMA-3.2-3B     | SPEAR        | `configs/math_mcqa_llama3b_spear.yaml` |
| StrategyQA    | Qwen2.5-1.5B     | SPEAR        | `configs/strategyqa_qwen1.5b_spear.yaml` |
| StrategyQA    | LLaMA-3.2-3B     | SPEAR        | `configs/strategyqa_llama3b_spear.yaml` |

Baseline configs are in `configs/baselines/{grpo,opsd,rltf_sd,feedback_sft}/`.

---

## Repository structure

```
spear_fl/
├── configs/                  # YAML experiment configs
│   └── baselines/            # configs for baseline algorithms
├── spear_fl/
│   ├── train.py              # entry point: python -m spear_fl.train --config ...
│   ├── federated.py          # federated learning loop (partitioning, aggregation)
│   ├── spear.py              # SPEAR algorithm implementation
│   ├── algorithms.py         # dispatch for all algorithms
│   ├── config.py             # dataclass configs and YAML loader
│   ├── data.py               # dataset loading and client partitioning
│   ├── models.py             # model and tokenizer loading
│   ├── lora.py               # LoRA adapter setup
│   ├── inference.py          # batched generation utilities
│   ├── feedback.py           # feedback providers
│   ├── metrics.py            # evaluation metrics
│   ├── partition.py          # Dirichlet data partitioning
│   ├── prompting.py          # prompt formatting per dataset
│   ├── extractors.py         # answer extraction from model outputs
│   ├── utils.py              # general utilities
│   └── hf_utils.py           # HuggingFace environment helpers
└── requirements.txt
```

---

## Requirements

- Python ≥ 3.9
- CUDA-capable GPU (bf16 support recommended)
- See `requirements.txt` for Python package versions
