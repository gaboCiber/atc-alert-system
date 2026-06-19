from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple


@dataclass
class E3Config:
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    ground_truth_dir: Path = field(init=False)
    transcriptions_dir: Path = field(init=False)
    ground_truth_file: Optional[Path] = field(default=None, init=False)
    transcriptions_file: Optional[Path] = field(default=None, init=False)
    results_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    dataset_name: Optional[str] = field(default=None, init=False)
    auto_discover: bool = field(default=False, init=False)

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
        ground_truth_file: Optional[str] = None,
        transcriptions_file: Optional[str] = None,
        output_dir: Optional[str] = None,
        auto_discover: bool = False,
        dataset_name: Optional[str] = None,
    ):
        base = Path(base_dir) if base_dir else Path(__file__).parent.parent
        cfg = cls(base_dir=base)
        if ground_truth_dir:
            cfg.ground_truth_dir = Path(ground_truth_dir)
        if transcriptions_dir:
            cfg.transcriptions_dir = Path(transcriptions_dir)
        if ground_truth_file:
            cfg.ground_truth_file = Path(ground_truth_file)
        if transcriptions_file:
            cfg.transcriptions_file = Path(transcriptions_file)
        if output_dir:
            cfg.results_dir = Path(output_dir)
            cfg.figures_dir = cfg.results_dir / "figures"
        cfg.auto_discover = auto_discover
        cfg.dataset_name = dataset_name
        return cfg

    @staticmethod
    def discover_dataset_pairs(gt_dir: Path, trans_dir: Path) -> List[Tuple[str, Path, Path]]:
        """
        Find matching pairs: ground_truth_{name}.csv + transcription_{name}.csv
        Returns [(dataset_name, gt_path, trans_path), ...]
        """
        gt_files = {}
        for f in gt_dir.glob("ground_truth_*.csv"):
            name = f.stem.replace("ground_truth_", "")
            gt_files[name] = f
        
        trans_files = {}
        for f in trans_dir.glob("transcription_*.csv"):
            name = f.stem.replace("transcription_", "")
            trans_files[name] = f
        
        common = set(gt_files.keys()) & set(trans_files.keys())
        return [(name, gt_files[name], trans_files[name]) for name in sorted(common)]


@dataclass
class EvalConfig:
    use_atc_normalizer: bool = True
    use_jiwer: bool = True
    detailed: bool = True
    visualization_dpi: int = 150
    visualization_style: str = "seaborn-v0.8-darkgrid"
