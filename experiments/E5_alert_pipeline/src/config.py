from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal


@dataclass
class E5Config:
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    ground_truth_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    compiled_rules_dir: Path = field(init=False)

    def __post_init__(self):
        self.ground_truth_dir = self.base_dir / "ground_truth"
        self.models_dir = self.base_dir / "models"
        self.results_dir = self.base_dir / "results"
        self.figures_dir = self.results_dir / "figures"
        e4_base = self.base_dir.parent / "E4_compiled_rules"
        self.compiled_rules_dir = e4_base / "models" / "model_A(gemma4:e4b)"

    @classmethod
    def from_dirs(
        cls,
        base_dir: Optional[str] = None,
        ground_truth_dir: Optional[str] = None,
        models_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        compiled_rules_dir: Optional[str] = None,
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
        if compiled_rules_dir:
            cfg.compiled_rules_dir = Path(compiled_rules_dir)
        return cfg


@dataclass
class JudgeConfig:
    model_name: str = "gemma4:31b-cloud"
    provider: Literal["openai", "gemini", "anthropic"] = "openai"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    max_retries: int = 3
    timeout: int = 120
    enabled: bool = True
    skip_on_error: bool = True


@dataclass
class GenericConfig:
    model_name: str = "gemma4:31b-cloud"
    provider: Literal["openai", "gemini", "anthropic"] = "openai"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    max_retries: int = 3
    timeout: int = 120


@dataclass
class MetricConfig:
    alert_precision_weight: float = 0.30
    alert_recall_weight: float = 0.30
    severity_accuracy_weight: float = 0.20
    semantic_quality_weight: float = 0.20
    visualization_dpi: int = 150
