import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Set

from .config import E2Config


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


class HolisticCheckpointer:
    """Intra-page checkpointing for holistic_judge calls.

    Saves progress after each completed chunk so evaluation resumes
    from the last completed chunk, not from scratch, if interrupted.
    """
    def __init__(
        self, results_dir: Path, model_name: str, page: int, config_hash: str
    ):
        self.path = results_dir / "checkpoints" / f"holistic_intra_{model_name}_page{page}.json"
        self.model_name = model_name
        self.page = page
        self.config_hash = config_hash
        self.completed_chunks: Set[int] = set()
        self.chunk_data: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("config_hash") != self.config_hash:
                return
            self.completed_chunks = set(data.get("completed_chunks", []))
            self.chunk_data = {int(k): v for k, v in data.get("chunk_data", {}).items()}
        except (json.JSONDecodeError, KeyError, IOError):
            pass

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {
            "config_hash": self.config_hash,
            "model_name": self.model_name,
            "page": self.page,
            "completed_chunks": sorted(self.completed_chunks),
            "chunk_data": {str(k): v for k, v in self.chunk_data.items()},
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def is_chunk_completed(self, chunk_idx: int) -> bool:
        return chunk_idx in self.completed_chunks

    def mark_chunk_completed(
        self, chunk_idx: int, per_type_data: Dict[str, Dict[str, Any]]
    ):
        self.completed_chunks.add(chunk_idx)
        self.chunk_data[chunk_idx] = per_type_data
        self._save()

    def get_chunk_data(self, chunk_idx: int) -> Optional[Dict[str, Dict[str, Any]]]:
        return self.chunk_data.get(chunk_idx)

    def cleanup(self):
        if self.path.exists():
            self.path.unlink()