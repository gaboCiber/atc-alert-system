from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal


@dataclass
class E4Config:
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    ground_truth_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    judge_cache_dir: Optional[Path] = None

    def __post_init__(self):
        self.ground_truth_dir = self.base_dir / "ground_truth"
        self.models_dir = self.base_dir / "models"
        self.results_dir = self.base_dir / "results"
        self.figures_dir = self.results_dir / "figures"
        if self.judge_cache_dir is None:
            self.judge_cache_dir = self.results_dir

    @classmethod
    def from_dirs(
        cls,
        base_dir: Optional[str] = None,
        ground_truth_dir: Optional[str] = None,
        models_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        judge_cache_dir: Optional[str] = None,
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
        if judge_cache_dir:
            cfg.judge_cache_dir = Path(judge_cache_dir)
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
class MetricConfig:
    classification_weight: float = 0.15
    validation_weight: float = 0.15
    execution_weight: float = 0.30
    semantic_weight: float = 0.40

    visualization_dpi: int = 150