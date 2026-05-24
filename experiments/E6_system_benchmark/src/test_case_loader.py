import json
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
    eval_type: str = "native+compiled"


def load_test_cases(gt_dir: Path) -> List[TestCase]:
    path = gt_dir / "test_cases.json"
    if not path.exists():
        print(f"  WARNING: {path} not found")
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
                eval_type=item.get("eval_type", "native+compiled"),
            )
        )
    return cases
