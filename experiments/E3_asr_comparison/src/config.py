from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class E3Config:
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    ground_truth_dir: Path = field(init=False)
    transcriptions_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)

    def __post_init__(self):
        self.ground_truth_dir = self.base_dir / "ground_truth"
        self.transcriptions_dir = self.base_dir / "transcriptions"
        self.results_dir = self.base_dir / "results"
        self.figures_dir = self.results_dir / "figures"

    @classmethod
    def from_dirs(
        cls,
        base_dir: Optional[str] = None,
        ground_truth_dir: Optional[str] = None,
        transcriptions_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        base = Path(base_dir) if base_dir else Path(__file__).parent.parent
        cfg = cls(base_dir=base)
        if ground_truth_dir:
            cfg.ground_truth_dir = Path(ground_truth_dir)
        if transcriptions_dir:
            cfg.transcriptions_dir = Path(transcriptions_dir)
        if output_dir:
            cfg.results_dir = Path(output_dir)
            cfg.figures_dir = cfg.results_dir / "figures"
        return cfg


@dataclass
class EvalConfig:
    use_atc_normalizer: bool = True
    use_jiwer: bool = True
    detailed: bool = True
    visualization_dpi: int = 150
    visualization_style: str = "seaborn-v0.8-darkgrid"
