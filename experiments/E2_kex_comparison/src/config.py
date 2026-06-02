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
    provider: Literal["openai", "gemini", "anthropic"] = "openai"
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
    semantic_weight: float = 0.60

    fuzzy_threshold: float = 70.0
    visualization_dpi: int = 150
    visualization_style: str = "seaborn-v0.8-darkgrid"
