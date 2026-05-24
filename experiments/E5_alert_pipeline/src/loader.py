import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExpectedAlert:
    satisfied: bool
    severity: Optional[str] = None


@dataclass
class TestCase:
    id: str
    instruction: str
    callsign: Optional[str]
    description: str
    traffic_state: dict
    expected_alerts: Dict[str, ExpectedAlert]


@dataclass
class CompiledRule:
    rule_id: str
    evaluate_fn: Any


@dataclass
class ExperimentData:
    test_cases: List[TestCase] = field(default_factory=list)
    rule_descriptions: Dict[str, str] = field(default_factory=dict)
    compiled_rules: Dict[str, CompiledRule] = field(default_factory=dict)
    strategy_names: List[str] = field(default_factory=lambda: ["compiled", "generic"])
    ground_truth_classification: Dict[str, bool] = field(default_factory=dict)

    @property
    def relevant_rule_ids(self) -> List[str]:
        ids = set()
        for tc in self.test_cases:
            ids.update(tc.expected_alerts.keys())
        return sorted(ids)

    @classmethod
    def from_config(cls, cfg) -> "ExperimentData":
        data = cls()

        data.rule_descriptions = load_rule_descriptions(cfg.ground_truth_dir)
        test_cases = load_test_cases(cfg.ground_truth_dir)
        data.test_cases = test_cases

        data.compiled_rules = load_compiled_rules(cfg.compiled_rules_dir)

        data.ground_truth_classification = {}
        for rule_id in data.relevant_rule_ids:
            data.ground_truth_classification[rule_id] = True

        return data


def load_test_cases(gt_dir: Path) -> List[TestCase]:
    path = gt_dir / "test_cases.json"
    if not path.exists():
        print(f"  WARNING: {path} not found, no test cases")
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cases = []
    for item in raw:
        expected = {}
        for rid, val in item.get("expected_alerts", {}).items():
            expected[rid] = ExpectedAlert(
                satisfied=val.get("satisfied", True),
                severity=val.get("severity"),
            )
        cases.append(
            TestCase(
                id=item["id"],
                instruction=item.get("instruction", ""),
                callsign=item.get("callsign"),
                description=item.get("description", ""),
                traffic_state=item.get("traffic_state", {}),
                expected_alerts=expected,
            )
        )
    return cases


def load_rule_descriptions(gt_dir: Path) -> Dict[str, str]:
    path = gt_dir / "rule_descriptions.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_compiled_rules(rules_dir: Path) -> Dict[str, CompiledRule]:
    if not rules_dir.exists():
        print(f"  WARNING: compiled rules dir {rules_dir} not found")
        return {}
    result = {}
    for py_file in rules_dir.glob("RULE*.py"):
        rule_id = py_file.stem
        try:
            spec = importlib.util.spec_from_file_location(rule_id, py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[rule_id] = module
                spec.loader.exec_module(module)
                if hasattr(module, "evaluate"):
                    result[rule_id] = CompiledRule(
                        rule_id=rule_id, evaluate_fn=module.evaluate
                    )
                else:
                    print(f"  WARNING: {py_file.name} has no evaluate() function")
        except Exception as e:
            print(f"  WARNING: failed to load {py_file.name}: {e}")
    return result
