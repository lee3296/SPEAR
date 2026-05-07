from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .config import HuggingFaceConfig


def apply_hf_settings(hf: HuggingFaceConfig) -> Dict[str, Any]:
    """Apply environment variables and return common kwargs for HF loaders.

    - Avoids repeated network metadata calls by enabling offline mode when requested.
    - Allows configuring cache locations for models and datasets.
    """
    if hf.hf_home:
        os.environ["HF_HOME"] = hf.hf_home

    # Hub cache (snapshots / metadata)
    hub_cache = hf.hub_cache_dir or hf.model_cache_dir
    if hub_cache:
        os.environ["HF_HUB_CACHE"] = hub_cache

    # Transformers cache (models/tokenizers)
    if hf.model_cache_dir:
        # TRANSFORMERS_CACHE is legacy but still widely respected.
        os.environ["TRANSFORMERS_CACHE"] = hf.model_cache_dir

    # Datasets cache
    if hf.dataset_cache_dir:
        os.environ["HF_DATASETS_CACHE"] = hf.dataset_cache_dir

    if hf.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    return {
        "model_cache_dir": hf.model_cache_dir,
        "dataset_cache_dir": hf.dataset_cache_dir,
        "hub_cache_dir": hub_cache,
        "local_files_only": bool(hf.offline),
    }


def merge_hf_overrides(cfg_hf: HuggingFaceConfig, *,
                      offline: Optional[bool] = None,
                      hf_home: Optional[str] = None,
                      model_cache_dir: Optional[str] = None,
                      dataset_cache_dir: Optional[str] = None,
                      hub_cache_dir: Optional[str] = None) -> HuggingFaceConfig:
    """Return a copy of cfg_hf with CLI overrides applied (None = no change)."""
    out = HuggingFaceConfig(
        offline=cfg_hf.offline,
        hf_home=cfg_hf.hf_home,
        model_cache_dir=cfg_hf.model_cache_dir,
        dataset_cache_dir=cfg_hf.dataset_cache_dir,
        hub_cache_dir=cfg_hf.hub_cache_dir,
    )
    if offline is not None:
        out.offline = offline
    if hf_home is not None:
        out.hf_home = hf_home
    if model_cache_dir is not None:
        out.model_cache_dir = model_cache_dir
    if dataset_cache_dir is not None:
        out.dataset_cache_dir = dataset_cache_dir
    if hub_cache_dir is not None:
        out.hub_cache_dir = hub_cache_dir
    return out
