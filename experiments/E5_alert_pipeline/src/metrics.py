from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loader import TestCase, ExpectedAlert
from pipeline_runner import AlertResult, StrategyResult


@dataclass
class RuleMetrics:
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    severity_correct: int = 0
    severity_total: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        denom = p + r
        return 2 * p * r / denom if denom > 0 else 0.0

    @property
    def accuracy(self) -> float:
        denom = (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
        )
        correct = self.true_positives + self.true_negatives
        return correct / denom if denom > 0 else 0.0

    @property
    def severity_accuracy(self) -> float:
        return (
            self.severity_correct / self.severity_total
            if self.severity_total > 0
            else 0.0
        )


@dataclass
class StrategyMetrics:
    strategy_name: str
    per_rule: Dict[str, RuleMetrics] = field(default_factory=dict)
    avg_latency_ms: float = 0.0

    @property
    def overall_precision(self) -> float:
        vals = [m.precision for m in self.per_rule.values()]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def overall_recall(self) -> float:
        vals = [m.recall for m in self.per_rule.values()]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def overall_f1(self) -> float:
        vals = [m.f1 for m in self.per_rule.values()]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def overall_accuracy(self) -> float:
        vals = [m.accuracy for m in self.per_rule.values()]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def overall_severity_accuracy(self) -> float:
        vals = [m.severity_accuracy for m in self.per_rule.values()]
        return sum(vals) / len(vals) if vals else 0.0


def calculate_metrics(
    strategy_result: StrategyResult,
    test_cases: List[TestCase],
) -> StrategyMetrics:
    metrics = StrategyMetrics(strategy_name=strategy_result.strategy_name)

    all_rule_ids = set()
    for tc in test_cases:
        all_rule_ids.update(tc.expected_alerts.keys())

    for rule_id in sorted(all_rule_ids):
        rm = RuleMetrics()
        for tc in test_cases:
            expected = tc.expected_alerts.get(rule_id)
            if expected is None:
                continue
            actual = strategy_result.per_test_case.get(tc.id, {}).get(rule_id)
            if actual is None:
                continue

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

        metrics.per_rule[rule_id] = rm

    metrics.avg_latency_ms = strategy_result.total_latency_ms
    return metrics
