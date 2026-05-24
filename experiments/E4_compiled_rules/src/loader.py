import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config import E4Config


@dataclass
class CompiledRule:
    rule_id: str
    model_name: str
    category: str
    compiled_code: str
    is_compilable: bool
    compilation_status: str
    failure_reason: Optional[str] = None
    compilation_metadata: Optional[dict] = None


@dataclass
class ClassificationGT:
    is_compilable: bool
    reason: str


@dataclass
class TestTrafficState:
    name: str
    state: dict
    expected_outcome: Optional[dict] = None


@dataclass
class ModelCompilationResult:
    model_name: str
    rule_ids: List[str]
    compiled_rules: Dict[str, CompiledRule] = field(default_factory=dict)
    failed_rules: Dict[str, str] = field(default_factory=dict)


def load_ground_truth_classification(gt_dir: Path) -> Dict[str, ClassificationGT]:
    path = gt_dir / "expected_classification.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: ClassificationGT(**v) for k, v in data.items()}


def load_reference_code(gt_dir: Path) -> Dict[str, str]:
    ref_dir = gt_dir / "reference_code"
    if not ref_dir.exists():
        return {}
    result = {}
    for f in ref_dir.glob("*.py"):
        with open(f, "r", encoding="utf-8") as fh:
            result[f.stem] = fh.read()
    return result


def load_test_traffic_states(gt_dir: Path) -> Dict[str, TestTrafficState]:
    ts_dir = gt_dir / "test_traffic_states"
    if not ts_dir.exists():
        return {}
    result = {}
    for f in ts_dir.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        name = f.stem
        expected = data.pop("expected_outcome", None)
        result[name] = TestTrafficState(name=name, state=data, expected_outcome=expected)
    return result


def discover_model_dirs(models_dir: Path) -> List[Path]:
    if not models_dir.exists():
        return []
    return sorted([d for d in models_dir.iterdir() if d.is_dir()], key=lambda x: x.name)


def load_manifest(model_dir: Path) -> Optional[dict]:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_compilation(model_dir: Path) -> ModelCompilationResult:
    manifest = load_manifest(model_dir)
    model_name = model_dir.name
    compiled_rules = {}
    failed_rules = {}

    if manifest and "rules" in manifest:
        for rule_id, rule_data in manifest["rules"].items():
            status = rule_data.get("compilation_status", "unknown")
            if status == "compiled":
                py_path = model_dir / f"{rule_id}.py"
                code = py_path.read_text() if py_path.exists() else rule_data.get("compiled_code", "")
                compiled_rules[rule_id] = CompiledRule(
                    rule_id=rule_id,
                    model_name=model_name,
                    category=rule_data.get("rule_category", "UNKNOWN"),
                    compiled_code=code,
                    is_compilable=True,
                    compilation_status=status,
                    compilation_metadata=rule_data.get("compilation_metadata"),
                )
            else:
                failed_rules[rule_id] = status

    return ModelCompilationResult(
        model_name=model_name,
        rule_ids=sorted(compiled_rules.keys()),
        compiled_rules=compiled_rules,
        failed_rules=failed_rules,
    )


@dataclass
class ExperimentData:
    model_names: List[str]
    model_results: Dict[str, ModelCompilationResult]
    ground_truth_classification: Dict[str, ClassificationGT]
    reference_code: Dict[str, str]
    test_traffic_states: Dict[str, TestTrafficState]

    @classmethod
    def from_config(cls, cfg: E4Config) -> "ExperimentData":
        model_dirs = discover_model_dirs(cfg.models_dir)
        if not model_dirs:
            raise FileNotFoundError(f"No model directories found in {cfg.models_dir}")

        model_results = {}
        all_rule_ids = set()

        for mdir in model_dirs:
            result = load_model_compilation(mdir)
            model_results[mdir.name] = result
            all_rule_ids.update(result.rule_ids)
            all_rule_ids.update(result.failed_rules.keys())

        gt_classification = load_ground_truth_classification(cfg.ground_truth_dir)
        ref_code = load_reference_code(cfg.ground_truth_dir)
        test_states = load_test_traffic_states(cfg.ground_truth_dir)

        return cls(
            model_names=sorted(model_results.keys()),
            model_results=model_results,
            ground_truth_classification=gt_classification,
            reference_code=ref_code,
            test_traffic_states=test_states,
        )