from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from test_case_loader import TestCase, ExpectedAlert
from benchmark_runner import EvalResult


@dataclass
class LatencyStats:
    step_name: str
    values_ms: List[float] = field(default_factory=list)

    def add(self, val_ms: float):
        if val_ms > 0:
            self.values_ms.append(val_ms)

    @property
    def count(self) -> int:
        return len(self.values_ms)

    @property
    def avg_ms(self) -> float:
        return sum(self.values_ms) / self.count if self.count else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.values_ms) if self.count else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.values_ms) if self.count else 0.0

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

    @property
    def p99_ms(self) -> float:
        if not self.count:
            return 0.0
        s = sorted(self.values_ms)
        idx = int(len(s) * 0.99)
        return s[min(idx, len(s) - 1)]

    def to_dict(self) -> Dict[str, float]:
        return {
            "avg_ms": round(self.avg_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "count": self.count,
        }


@dataclass
class AccuracyMetrics:
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    severity_correct: int = 0
    severity_total: int = 0

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
        p = self.precision
        r = self.recall
        d = p + r
        return 2 * p * r / d if d else 0.0

    @property
    def severity_accuracy(self) -> float:
        return self.severity_correct / self.severity_total if self.severity_total else 0.0


def aggregate_latency(all_benchmarks: List[Any]) -> Dict[str, LatencyStats]:
    steps = {}

    for suffix in ["bert", "native", "compiled", "generic", "pipeline"]:
        steps[suffix] = LatencyStats(step_name=suffix)

    for bench in all_benchmarks:
        if getattr(bench, "bert_ms", 0) > 0:
            steps["bert"].add(bench.bert_ms)
        for rule_id, ms in getattr(bench, "native_rules", {}).items():
            key = f"native_{rule_id}"
            if key not in steps:
                steps[key] = LatencyStats(step_name=key)
            steps[key].add(ms)
            steps["native"].add(ms)
        for rule_id, ms in getattr(bench, "compiled_rules", {}).items():
            key = f"compiled_{rule_id}"
            if key not in steps:
                steps[key] = LatencyStats(step_name=key)
            steps[key].add(ms)
            steps["compiled"].add(ms)
        for rule_id, ms in getattr(bench, "generic_rules", {}).items():
            key = f"generic_{rule_id}"
            if key not in steps:
                steps[key] = LatencyStats(step_name=key)
            steps[key].add(ms)
            steps["generic"].add(ms)
        if getattr(bench, "pipeline_e2e_ms", 0) > 0:
            steps["pipeline"].add(bench.pipeline_e2e_ms)

    return steps


def aggregate_accuracy(
    all_benchmarks: List[Any],
    test_cases: List[TestCase],
) -> Dict[str, AccuracyMetrics]:
    per_rule: Dict[str, AccuracyMetrics] = {}

    for tc in test_cases:
        bench = next((b for b in all_benchmarks if b.test_case_id == tc.id), None)
        if bench is None:
            continue

        for rule_id in tc.expected_alerts.keys():
            if rule_id not in per_rule:
                per_rule[rule_id] = AccuracyMetrics()

            expected = tc.expected_alerts[rule_id]
            actual = bench.eval_results.get(rule_id)
            if actual is None:
                continue

            rm = per_rule[rule_id]
            if expected.satisfied:
                if actual.satisfied:
                    rm.true_negatives += 1
                else:
                    rm.false_positives += 1
            else:
                if not actual.satisfied:
                    rm.true_positives += 1
                else:
                    rm.false_negatives += 1

            if not expected.satisfied and not actual.satisfied:
                rm.severity_total += 1
                if expected.severity and actual.severity:
                    if actual.severity.lower() == expected.severity.lower():
                        rm.severity_correct += 1

    return per_rule
