"""Training entry point."""
from __future__ import annotations

import argparse
import os

from .config import load_config
from .federated import run_federated
from .hf_utils import merge_hf_overrides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--hf_offline", action="store_true")
    ap.add_argument("--hf_home", type=str, default=None)
    ap.add_argument("--hf_model_cache_dir", type=str, default=None)
    ap.add_argument("--hf_dataset_cache_dir", type=str, default=None)
    ap.add_argument("--hf_hub_cache_dir", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.hf = merge_hf_overrides(
        cfg.hf,
        offline=True if args.hf_offline else None,
        hf_home=args.hf_home,
        model_cache_dir=args.hf_model_cache_dir,
        dataset_cache_dir=args.hf_dataset_cache_dir,
        hub_cache_dir=args.hf_hub_cache_dir,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    out = run_federated(cfg)
    print(f"Done. Outputs saved to: {out}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
