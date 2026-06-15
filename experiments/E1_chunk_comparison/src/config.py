from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class E1Config:
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    ground_truth_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    sentences_gt_path: Path = field(init=False)

    def __post_init__(self):
        self.ground_truth_dir = self.base_dir / "ground_truth"
        self.models_dir = self.base_dir / "models"
        self.results_dir = self.base_dir / "results"
        self.figures_dir = self.results_dir / "figures"
        self.sentences_gt_path = self.base_dir / "sentences_gt.json"

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