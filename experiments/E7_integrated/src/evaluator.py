"""Orquestador del benchmark E7 con resume por test case."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import E7Config, LLMConfig
from test_case_loader import TestCase, load_test_cases
from pipeline_benchmark import IntegratedPipelineBenchmark, TcPipelineResult, GeneratedAlertInfo
from metrics import aggregate_latency, aggregate_accuracy, evaluate_tc_accuracy
from judge import PipelineJudge, run_judge_all


@dataclass
class E7Results:
    config: Any = None
    all_results: List[TcPipelineResult] = field(default_factory=list)
    per_tc_accuracy: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    aggregate_accuracy: Dict[str, Any] = field(default_factory=dict)
    latency: Dict[str, Any] = field(default_factory=dict)
    judge_scores: Dict[str, float] = field(default_factory=dict)


def _serialize_result(r: TcPipelineResult) -> dict:
    return {
        "test_case_id": r.test_case_id,
        "total_ms": r.total_ms,
        "parse_ms": r.parse_ms,
        "steps": {str(k): v for k, v in r.steps.items()},
        "generated_alerts": [
            {
                "rule_id": a.rule_id,
                "severity": a.severity,
                "explanation": a.explanation,
                "condition_type": a.condition_type,
            }
            for a in r.generated_alerts
        ],
        "final_decision": r.final_decision,
        "has_errors": r.has_errors,
        "error": r.error,
    }


def _deserialize_result(d: dict) -> TcPipelineResult:
    alerts = [
        GeneratedAlertInfo(
            rule_id=a["rule_id"],
            severity=a.get("severity", ""),
            explanation=a.get("explanation", ""),
            condition_type=a.get("condition_type", ""),
        )
        for a in d.get("generated_alerts", [])
    ]
    return TcPipelineResult(
        test_case_id=d["test_case_id"],
        total_ms=d.get("total_ms", 0.0),
        parse_ms=d.get("parse_ms", 0.0),
        steps={int(k): v for k, v in d.get("steps", {}).items()},
        generated_alerts=alerts,
        final_decision=d.get("final_decision", "PENDING"),
        has_errors=d.get("has_errors", False),
        error=d.get("error"),
    )


def load_partial(results_dir: Path, filename: str) -> Dict[str, TcPipelineResult]:
    path = results_dir / filename
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {tid: _deserialize_result(d) for tid, d in raw.items()}
    except Exception:
        return {}


def save_partial(results_dir: Path, filename: str, partial: Dict[str, TcPipelineResult]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / filename
    data = {tid: _serialize_result(r) for tid, r in partial.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_benchmark(
    cfg: E7Config,
    judge: Optional[PipelineJudge] = None,
    tc_ids: Optional[List[str]] = None,
) -> E7Results:
    test_cases = load_test_cases(cfg.ground_truth_dir)
    if tc_ids:
        wanted = set(tc_ids)
        test_cases = [tc for tc in test_cases if tc.id in wanted]
        missing = wanted - {tc.id for tc in test_cases}
        if missing:
            print(f"  WARNING: TCs no encontrados: {', '.join(sorted(missing))}")
        print(f"  Subset: {len(test_cases)} test case(s)")
    benchmark = IntegratedPipelineBenchmark(cfg)

    partial = load_partial(cfg.results_dir, cfg.partial_file)
    if partial:
        print(f"  Resuming: {len(partial)} test cases already completed")
    print()

    all_results: List[TcPipelineResult] = []

    for tc in test_cases:
        if tc.id in partial:
            print(f"  [{tc.id}] [RESUMED]")
            all_results.append(partial[tc.id])
            continue

        print(f"  [{tc.id}] {tc.description}")
        result = benchmark.run_test_case(tc)

        if result.error:
            print(f"    ERROR: {result.error}")
        else:
            acc = evaluate_tc_accuracy(tc, result)
            gen_ids = [a.rule_id for a in result.generated_alerts]
            print(
                f"    E2E={result.total_ms:.0f}ms | Parse={result.parse_ms:.0f}ms | "
                f"Alerts={gen_ids or 'none'} | "
                f"TP={acc.true_positives} FP={acc.false_positives} FN={acc.false_negatives}"
            )

        all_results.append(result)
        partial[tc.id] = result
        save_partial(cfg.results_dir, cfg.partial_file, partial)

    print()

    latency_stats = aggregate_latency(all_results)
    accuracy = aggregate_accuracy(test_cases, all_results)

    e7 = E7Results(config=cfg, all_results=all_results)
    e7.latency = {k: v.to_dict() for k, v in latency_stats.items()}
    e7.per_tc_accuracy = accuracy["per_test_case"]
    e7.aggregate_accuracy = accuracy["aggregate"]

    if judge:
        e7.judge_scores = run_judge_all(judge, test_cases, all_results)

    _save_detailed(cfg, e7)
    return e7


def _save_detailed(cfg: E7Config, results: E7Results) -> None:
    detailed = {
        "results": [_serialize_result(r) for r in results.all_results],
        "per_tc_accuracy": results.per_tc_accuracy,
        "judge_scores": results.judge_scores,
    }
    path = cfg.results_dir / "detailed_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2)
