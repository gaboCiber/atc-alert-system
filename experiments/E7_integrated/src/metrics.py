"""Métricas de latencia y accuracy a nivel pipeline para E7."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from test_case_loader import TestCase
from pipeline_benchmark import TcPipelineResult, GeneratedAlertInfo


@dataclass
class LatencyStats:
    step_name: str
    values_ms: List[float] = field(default_factory=list)

    def add(self, val_ms: float) -> None:
        if val_ms > 0:
            self.values_ms.append(val_ms)

    @property
    def count(self) -> int:
        return len(self.values_ms)

    @property
    def avg_ms(self) -> float:
        return sum(self.values_ms) / self.count if self.count else 0.0

    @property
    def p50_ms(self) -> float:
        if not self.count:
            return 0.0
        s = sorted(self.values_ms)
        return s[len(s) // 2]

    @property
    def p95_ms(self) -> float:
        if not self.count:
            return 0.0
        s = sorted(self.values_ms)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    def to_dict(self) -> Dict[str, float]:
        return {
            "avg_ms": round(self.avg_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "count": self.count,
        }


@dataclass
class TcAccuracy:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    severity_correct: int = 0
    severity_total: int = 0
    expected_violations: List[str] = field(default_factory=list)
    generated_violations: List[str] = field(default_factory=list)
    unexpected_alerts: List[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def severity_accuracy(self) -> float:
        return self.severity_correct / self.severity_total if self.severity_total else 0.0


def evaluate_tc_accuracy(tc: TestCase, result: TcPipelineResult) -> TcAccuracy:
    """Evalúa accuracy pipeline-level para un test case."""
    acc = TcAccuracy()

    expected_violations = {
        rid for rid, exp in tc.expected_alerts.items() if not exp.satisfied
    }
    expected_safe = {
        rid for rid, exp in tc.expected_alerts.items() if exp.satisfied
    }

    generated_by_id: Dict[str, GeneratedAlertInfo] = {
        a.rule_id: a for a in result.generated_alerts
    }
    generated_ids = set(generated_by_id.keys())

    acc.expected_violations = sorted(expected_violations)
    acc.generated_violations = sorted(generated_ids)

    for rid in expected_violations:
        if rid in generated_ids:
            acc.true_positives += 1
            exp = tc.expected_alerts[rid]
            gen = generated_by_id[rid]
            acc.severity_total += 1
            if exp.severity and gen.severity.lower() == exp.severity.lower():
                acc.severity_correct += 1
        else:
            acc.false_negatives += 1

    if not tc.expected_alerts:
        acc.false_positives = len(generated_ids)
        acc.unexpected_alerts = sorted(generated_ids)
    else:
        unexpected = generated_ids - expected_violations
        for rid in unexpected:
            acc.false_positives += 1
            acc.unexpected_alerts.append(rid)
        for rid in expected_safe:
            if rid in generated_ids:
                acc.false_positives += 1
                if rid not in acc.unexpected_alerts:
                    acc.unexpected_alerts.append(rid)

    acc.unexpected_alerts = sorted(set(acc.unexpected_alerts))
    return acc


def aggregate_latency(results: List[TcPipelineResult]) -> Dict[str, LatencyStats]:
    stats = {
        "e2e": LatencyStats("e2e"),
        "parse": LatencyStats("parse"),
    }
    step_names = [
        "INPUT_PROCESSING",
        "NORMALIZATION",
        "STATE_PROJECTION",
        "RULE_EVALUATION",
        "ALERT_GENERATION",
        "ALERT_PRESENTATION",
        "ATCO_DECISION",
        "FINAL_STATE_UPDATE",
    ]
    for name in step_names:
        stats[name] = LatencyStats(name)

    for res in results:
        stats["e2e"].add(res.total_ms)
        stats["parse"].add(res.parse_ms)
        for step_info in res.steps.values():
            name = step_info.get("name", "")
            ms = step_info.get("execution_time_ms", 0.0)
            if name in stats:
                stats[name].add(ms)

    return stats


def aggregate_accuracy(
    test_cases: List[TestCase],
    results: List[TcPipelineResult],
) -> Dict[str, Any]:
    per_tc: Dict[str, Dict[str, Any]] = {}
    totals = TcAccuracy()

    for tc in test_cases:
        bench = next((r for r in results if r.test_case_id == tc.id), None)
        if bench is None:
            continue
        acc = evaluate_tc_accuracy(tc, bench)
        per_tc[tc.id] = {
            "precision": round(acc.precision, 4),
            "recall": round(acc.recall, 4),
            "f1": round(acc.f1, 4),
            "severity_accuracy": round(acc.severity_accuracy, 4),
            "tp": acc.true_positives,
            "fp": acc.false_positives,
            "fn": acc.false_negatives,
            "expected_violations": acc.expected_violations,
            "generated_violations": acc.generated_violations,
            "unexpected_alerts": acc.unexpected_alerts,
        }
        totals.true_positives += acc.true_positives
        totals.false_positives += acc.false_positives
        totals.false_negatives += acc.false_negatives
        totals.severity_correct += acc.severity_correct
        totals.severity_total += acc.severity_total

    return {
        "per_test_case": per_tc,
        "aggregate": {
            "precision": round(totals.precision, 4),
            "recall": round(totals.recall, 4),
            "f1": round(totals.f1, 4),
            "severity_accuracy": round(totals.severity_accuracy, 4),
            "tp": totals.true_positives,
            "fp": totals.false_positives,
            "fn": totals.false_negatives,
        },
    }
