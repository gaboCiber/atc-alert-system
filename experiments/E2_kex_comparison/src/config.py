from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal


@dataclass
class E2Config:
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    ground_truth_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)

    def __post_init__(self):
        self.ground_truth_dir = self.base_dir / "ground_truth"
        self.models_dir = self.base_dir / "models"
        self.results_dir = self.base_dir / "results"
        self.figures_dir = self.results_dir / "figures"

    @classmethod
    def from_dirs(
        cls,
        base_dir: Optional[str] = None,
        ground_truth_dir: Optional[str] = None,
        models_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        base = Path(base_dir) if base_dir else Path(__file__).parent.parent
        cfg = cls(base_dir=base)
        if ground_truth_dir:
            cfg.ground_truth_dir = Path(ground_truth_dir)
        if models_dir:
            cfg.models_dir = Path(models_dir)
        if output_dir:
            cfg.results_dir = Path(output_dir)
            cfg.figures_dir = cfg.results_dir / "figures"
        return cfg


@dataclass
class JudgeConfig:
    model_name: str = "llama3.2"
    provider: Literal["openai", "gemini", "anthropic", "ollama"] = "openai"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    max_retries: int = 3
    timeout: int = 120
    enabled: bool = True
    skip_on_error: bool = True


@dataclass
class DedupConfig:
    enabled: bool = True
    batch_size: int = 10
    threshold: float = 0.80


@dataclass
class MetricConfig:
    structural_weight: float = 0.15
    content_weight: float = 0.10
    cross_ref_weight: float = 0.15
    dedup_weight: float = 0.20
    semantic_weight: float = 0.55
    error_weight: float = 0.05

    fuzzy_threshold: float = 70.0
    visualization_dpi: int = 150
    visualization_style: str = "seaborn-v0.8-darkgrid"


def get_config_hash(
    judge_config: JudgeConfig,
    metric_config: MetricConfig,
    dedup_config: Optional[DedupConfig] = None,
) -> str:
    """
    Compute a hash of the configuration that affects the LLM evaluation results.
    Only judge config and dedup config affect LLM calls;
    metric weights are excluded so changing them doesn't invalidate checkpoints.
    """
    import hashlib
    import json

    judge_dict = {
        "model_name": judge_config.model_name,
        "provider": judge_config.provider,
        "base_url": judge_config.base_url,
        "api_key": judge_config.api_key,
        "max_retries": judge_config.max_retries,
        "timeout": judge_config.timeout,
        "enabled": judge_config.enabled,
        "skip_on_error": judge_config.skip_on_error,
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
        "dedup": dedup_dict,
    }
    config_json = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_json.encode()).hexdigest()
