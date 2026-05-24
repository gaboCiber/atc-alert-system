import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

from config import E3Config


@dataclass
class TranscriptionSample:
    model_name: str
    sample_id: str
    reference: str
    hypothesis: str


@dataclass
class ModelResult:
    model_name: str
    samples: List[TranscriptionSample] = field(default_factory=list)


def load_ground_truth_csv(gt_path: Path) -> Dict[str, str]:
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    if gt_path.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(gt_path)
        id_col = df.columns[0]
        text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        return dict(zip(df[id_col].astype(str), df[text_col].astype(str)))

    elif gt_path.suffix == ".json":
        with open(gt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {item.get("id", str(i)): item.get("text", "") for i, item in enumerate(data)}

    elif gt_path.suffix in (".txt",):
        result = {}
        with open(gt_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    result[f"sample_{i:04d}"] = line
        return result

    raise ValueError(f"Unsupported ground truth format: {gt_path.suffix}")


def load_transcriptions_csv(csv_path: Path) -> Dict[str, Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Transcriptions not found: {csv_path}")

    import pandas as pd
    df = pd.read_csv(csv_path, index_col=0)
    df.columns = pd.Index(pd.Series(df.columns).apply(lambda x: Path(str(x)).stem))

    results = {}
    for model in df.index:
        results[str(model)] = {str(k): str(v) if pd.notna(v) else "" for k, v in df.loc[model].items()}

    return results


def discover_transcription_files(transcriptions_dir: Path) -> List[Path]:
    if not transcriptions_dir.exists():
        return []
    return sorted(transcriptions_dir.glob("*.csv"))


def load_experiment_data(cfg: E3Config) -> tuple[
    List[str],
    Dict[str, Dict[str, str]],
    Dict[str, str],
]:
    gt_files = list(cfg.ground_truth_dir.glob("*"))
    if not gt_files:
        raise FileNotFoundError(f"No ground truth files found in {cfg.ground_truth_dir}")

    gt_path = gt_files[0]
    ground_truth = load_ground_truth_csv(gt_path)

    csv_files = discover_transcription_files(cfg.transcriptions_dir)
    if not csv_files:
        raise FileNotFoundError(f"No transcription CSV files found in {cfg.transcriptions_dir}")

    all_transcriptions: Dict[str, Dict[str, str]] = {}
    for csv_file in csv_files:
        model_transcriptions = load_transcriptions_csv(csv_file)
        all_transcriptions.update(model_transcriptions)

    model_names = sorted(all_transcriptions.keys())
    if not model_names:
        raise ValueError("No models found in transcription files")

    return model_names, all_transcriptions, ground_truth


@dataclass
class ExperimentData:
    model_names: List[str]
    all_transcriptions: Dict[str, Dict[str, str]]
    ground_truth: Dict[str, str]
    common_ids: List[str]

    @classmethod
    def from_config(cls, cfg: E3Config) -> "ExperimentData":
        model_names, all_transcriptions, ground_truth = load_experiment_data(cfg)

        gt_ids = set(ground_truth.keys())
        all_model_ids = set()
        for model_transcriptions in all_transcriptions.values():
            all_model_ids.update(model_transcriptions.keys())

        common_ids = sorted(gt_ids & all_model_ids)
        if not common_ids:
            raise ValueError("No common IDs between ground truth and transcriptions")

        return cls(
            model_names=model_names,
            all_transcriptions=all_transcriptions,
            ground_truth=ground_truth,
            common_ids=common_ids,
        )

    def get_model_transcriptions(self, model_name: str) -> Dict[str, str]:
        return self.all_transcriptions.get(model_name, {})
