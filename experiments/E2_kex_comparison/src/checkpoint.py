import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .config import E2Config, JudgeConfig, MetricConfig, DedupConfig, get_config_hash


def get_config_hash(
    judge_config: JudgeConfig,
    metric_config: MetricConfig,
    dedup_config: Optional[DedupConfig] = None,
) -> str:
    """
    Compute a hash of the configuration that affects the evaluation results.
    We include the judge config, metric config, and dedup config (if provided).
    """
    # We'll convert the configs to dictionaries and then to a JSON string for hashing.
    # We exclude fields that are not relevant to the computation (like paths, etc.)
    judge_dict = {
        "model_name": judge_config.model_name,
        "provider": judge_config.provider,
        # base_url and api_key might change but they don't affect the result if the model is the same.
        # However, if the base_url points to a different model, we should include it.
        # For simplicity, we include base_url and api_key as they might affect which model is used.
        "base_url": judge_config.base_url,
        "api_key": judge_config.api_key,
        "max_retries": judge_config.max_retries,
        "timeout": judge_config.timeout,
        "enabled": judge_config.enabled,
        "skip_on_error": judge_config.skip_on_error,
    }
    metric_dict = {
        "structural_weight": metric_config.structural_weight,
        "content_weight": metric_config.content_weight,
        "cross_ref_weight": metric_config.cross_ref_weight,
        "dedup_weight": metric_config.dedup_weight,
        "semantic_weight": metric_config.semantic_weight,
        "fuzzy_threshold": metric_config.fuzzy_threshold,
        # We exclude visualization settings as they don't affect the computed scores.
    }
    dedup_dict = {}
    if dedup_config is not None:
        dedup_dict = {
            "enabled": dedup_config.enabled,
            "batch_size": dedup_config.batch_size,
            "threshold": dedup_config.threshold,
        }

    config_dict = {
        "judge": judge_dict,
        "metric": metric_dict,
        "dedup": dedup_dict,
    }
    config_json = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_json.encode()).hexdigest()


def ensure_checkpoint_dir(results_dir: Path) -> Path:
    """
    Ensure the checkpoint directory exists and return its path.
    """
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_checkpoint(
    results_dir: Path,
    prefix: str,
    identifier: str,
    data: Dict[str, Any],
    config_hash: str,
) -> None:
    """
    Save a checkpoint to the results/checkpoints directory.
    The checkpoint file is named: {prefix}_{identifier}.json
    The data stored includes the actual data and the config_hash for validation.
    """
    checkpoint_dir = ensure_checkpoint_dir(results_dir)
    checkpoint_file = checkpoint_dir / f"{prefix}_{identifier}.json"
    to_save = {
        "config_hash": config_hash,
        "data": data,
    }
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=2)


def load_checkpoint(
    results_dir: Path,
    prefix: str,
    identifier: str,
    expected_config_hash: str,
) -> Optional[Dict[str, Any]]:
    """
    Load a checkpoint if it exists and the config hash matches.
    Returns the data if valid, otherwise None.
    """
    checkpoint_dir = results_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    checkpoint_file = checkpoint_dir / f"{prefix}_{identifier}.json"
    if not checkpoint_file.exists():
        return None
    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        if checkpoint.get("config_hash") == expected_config_hash:
            return checkpoint.get("data")
        else:
            # Config hash mismatch, we should not use this checkpoint.
            return None
    except (json.JSONDecodeError, KeyError, IOError):
        # If there's any error in reading, we ignore the checkpoint.
        return None