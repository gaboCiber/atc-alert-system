from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Literal


@dataclass
class E6Config:
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    ground_truth_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    compiled_rules_dir: Path = field(init=False)

    warmup_iterations: int = 2
    measure_iterations: int = 15

    def __post_init__(self):
        self.ground_truth_dir = self.base_dir / "ground_truth"
        self.results_dir = self.base_dir / "results"
        self.figures_dir = self.results_dir / "figures"
        e4_base = self.base_dir.parent / "E4_compiled_rules"
        self.compiled_rules_dir = e4_base / "models" / "gpt-oss:120b-cloud"

    @classmethod
    def from_dirs(
        cls,
        base_dir: Optional[str] = None,
        ground_truth_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        compiled_rules_dir: Optional[str] = None,
    ):
        base = Path(base_dir) if base_dir else Path(__file__).parent.parent
        cfg = cls(base_dir=base)
        if ground_truth_dir:
            cfg.ground_truth_dir = Path(ground_truth_dir)
        if output_dir:
            cfg.results_dir = Path(output_dir)
            cfg.figures_dir = cfg.results_dir / "figures"
        if compiled_rules_dir:
            cfg.compiled_rules_dir = Path(compiled_rules_dir)
        return cfg


@dataclass
class LLMConfig:
    model_name: str = "gemma4:31b-cloud"
    provider: Literal["openai", "gemini", "anthropic"] = "openai"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    max_retries: int = 3
    timeout: int = 120


@dataclass
class RuleDescription:
    rule_id: str
    description: str
    eval_type: str  # "native", "compiled", "generic"


@dataclass
class MetricConfig:
    latency_warning_ms: float = 500.0
    latency_critical_ms: float = 2000.0
    visualization_dpi: int = 150
