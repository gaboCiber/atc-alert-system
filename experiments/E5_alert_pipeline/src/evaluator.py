from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import E5Config, JudgeConfig, GenericConfig, MetricConfig
from loader import ExperimentData, TestCase
from pipeline_runner import run_strategy, StrategyResult
from semantic_judge import SemanticJudge, run_judge_evaluation
from metrics import calculate_metrics, StrategyMetrics


@dataclass
class E5Results:
    config: E5Config
    strategies: Dict[str, StrategyMetrics] = field(default_factory=dict)
    strategy_results: Dict[str, StrategyResult] = field(default_factory=dict)
    judge_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    test_cases: List[TestCase] = field(default_factory=list)
    overall_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def ranking(self) -> List[Dict[str, Any]]:
        ranked = sorted(
            self.overall_scores.items(), key=lambda x: x[1], reverse=True
        )
        result = []
        for rank, (name, score) in enumerate(ranked, 1):
            sm = self.strategies.get(name)
            js = self.judge_scores.get(name, {})
            sem_avg = sum(js.values()) / len(js) if js else 0.0
            entry = {
                "rank": rank,
                "strategy": name,
                "overall_score": round(score, 4),
                "alert_precision": round(sm.overall_precision, 4) if sm else 0.0,
                "alert_recall": round(sm.overall_recall, 4) if sm else 0.0,
                "alert_f1": round(sm.overall_f1, 4) if sm else 0.0,
                "severity_accuracy": round(sm.overall_severity_accuracy, 4) if sm else 0.0,
                "avg_latency_ms": round(sm.avg_latency_ms, 2) if sm else 0.0,
                "semantic_score": round(sem_avg, 4),
            }
            result.append(entry)
        return result


def run_evaluation(
    data: ExperimentData,
    judge: SemanticJudge,
    generic_cfg: GenericConfig,
    metric_cfg: MetricConfig,
) -> E5Results:
    results = E5Results(
        config=None,
        test_cases=data.test_cases,
    )

    strategy_results: Dict[str, StrategyResult] = {}

    for strategy in data.strategy_names:
        print(f"\n  Running: {strategy}")
        sr = run_strategy(
            strategy=strategy,
            test_cases=data.test_cases,
            compiled_rules=data.compiled_rules,
            rule_descriptions=data.rule_descriptions,
            generic_cfg=generic_cfg,
        )
        strategy_results[strategy] = sr

        print_cases = [
            f"    {tc.id}: "
            + ", ".join(
                f"{rid}={'✓' if alert.satisfied else '✗'}"
                for rid, alert in sr.per_test_case.get(tc.id, {}).items()
            )
            for tc in data.test_cases[:3]
        ]
        for line in print_cases:
            print(line)
        print(f"    Avg latency: {sr.total_latency_ms:.1f}ms")

    print(f"\n  Running semantic judge ({'enabled' if judge.cfg.enabled else 'disabled'})")
    judge_scores = run_judge_evaluation(judge, data.test_cases, strategy_results)

    strategies_metrics = {}
    for strategy, sr in strategy_results.items():
        strategies_metrics[strategy] = calculate_metrics(sr, data.test_cases)

    overall_scores = {}
    for strategy in data.strategy_names:
        sm = strategies_metrics[strategy]
        js = judge_scores.get(strategy, {})
        sem_avg = sum(js.values()) / len(js) if js else 0.0

        score = (
            metric_cfg.alert_precision_weight * sm.overall_precision
            + metric_cfg.alert_recall_weight * sm.overall_recall
            + metric_cfg.severity_accuracy_weight * sm.overall_severity_accuracy
            + metric_cfg.semantic_quality_weight * sem_avg
        )
        overall_scores[strategy] = score

    results.strategies = strategies_metrics
    results.strategy_results = strategy_results
    results.judge_scores = judge_scores
    results.overall_scores = overall_scores
    return results
