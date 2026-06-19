import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import E6Config, LLMConfig
from test_case_loader import TestCase, load_test_cases
import benchmark_runner
from benchmark_runner import SystemBenchmark, TcBenchmark, EvalResult
from metrics import aggregate_latency, aggregate_accuracy, LatencyStats, AccuracyMetrics
from semantic_judge import SemanticJudge, run_judge_all


PARTIAL_FILE = "partial_results.json"


def _serialize_bench(b: TcBenchmark) -> dict:
    return {
        "test_case_id": b.test_case_id,
        "bert_ms": b.bert_ms,
        "native_rules": b.native_rules,
        "compiled_rules": b.compiled_rules,
        "generic_rules": b.generic_rules,
        "pipeline_e2e_ms": b.pipeline_e2e_ms,
        "pipeline_steps": {str(k): v for k, v in b.pipeline_steps.items()},
        "eval_results": {
            rid: {
                "rule_id": r.rule_id,
                "satisfied": r.satisfied,
                "severity": r.severity,
                "latency_ms": r.latency_ms,
                "error": r.error,
                "explanation": r.explanation,
            }
            for rid, r in b.eval_results.items()
        },
    }


def _deserialize_bench(d: dict) -> TcBenchmark:
    eval_results = {}
    for rid, rd in d.get("eval_results", {}).items():
        eval_results[rid] = EvalResult(
            rule_id=rd["rule_id"],
            satisfied=rd["satisfied"],
            severity=rd["severity"],
            latency_ms=rd["latency_ms"],
            error=rd.get("error"),
            explanation=rd.get("explanation", ""),
        )
    return TcBenchmark(
        test_case_id=d["test_case_id"],
        bert_ms=d.get("bert_ms", 0.0),
        native_rules=d.get("native_rules", {}),
        compiled_rules=d.get("compiled_rules", {}),
        generic_rules=d.get("generic_rules", {}),
        pipeline_e2e_ms=d.get("pipeline_e2e_ms", 0.0),
        pipeline_steps={int(k): v for k, v in d.get("pipeline_steps", {}).items()},
        eval_results=eval_results,
    )


def load_partial(results_dir: Path) -> Dict[str, TcBenchmark]:
    path = results_dir / PARTIAL_FILE
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            raw = json.load(f)
        return {tid: _deserialize_bench(d) for tid, d in raw.items()}
    except Exception:
        return {}


def save_partial(results_dir: Path, partial: Dict[str, TcBenchmark]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / PARTIAL_FILE
    data = {tid: _serialize_bench(b) for tid, b in partial.items()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


@dataclass
class E6Results:
    config: Any = None
    all_benchmarks: List[TcBenchmark] = field(default_factory=list)
    latency: Dict[str, LatencyStats] = field(default_factory=dict)
    accuracy: Dict[str, AccuracyMetrics] = field(default_factory=dict)
    judge_scores: Dict[str, float] = field(default_factory=dict)
    pipeline_steps: Dict[str, List[Dict]] = field(default_factory=dict)
    ranking: List[Dict[str, Any]] = field(default_factory=list)


def run_benchmark(
    cfg: E6Config,
    llm_cfg: LLMConfig,
    judge: Optional[SemanticJudge],
    rule_descriptions: Dict[str, str],
) -> E6Results:
    test_cases = load_test_cases(cfg.ground_truth_dir)

    benchmark = SystemBenchmark(
        compiled_rules_dir=cfg.compiled_rules_dir,
        llm_cfg=llm_cfg,
        verbose=True,
    )

    print(f"\n  Loaded {len(benchmark.compiled_rules)} compiled rules")
    print(f"  Test cases: {len(test_cases)}")

    # Load partial results for resume
    partial = load_partial(cfg.results_dir)
    if partial:
        print(f"  Resuming: {len(partial)} test cases already completed")
    print()

    all_benchmarks: List[TcBenchmark] = []

    for tc in test_cases:
        # Resume: skip if already completed
        if tc.id in partial:
            print(f"  [{tc.id}] [RESUMED] (cached)")
            all_benchmarks.append(partial[tc.id])
            continue

        print(f"  [{tc.id}] {tc.description}")

        bert_times: List[float] = []
        native_times: Dict[str, List[float]] = {}
        compiled_times: Dict[str, List[float]] = {}
        generic_times: Dict[str, List[float]] = {}

        warmup = cfg.warmup_iterations
        measures = cfg.measure_iterations

        for iteration in range(warmup + measures):
            is_meas = iteration >= warmup

            # BERT (if available)
            bt = benchmark.benchmark_bert(tc.instruction)
            if bt > 0 and is_meas:
                bert_times.append(bt)

            # Native rules
            nt = benchmark.benchmark_native(tc.traffic_state, tc.callsign)
            if is_meas:
                for k, v in nt.items():
                    native_times.setdefault(k, []).append(v)

            # Compiled rules
            ct = benchmark.benchmark_compiled(tc.traffic_state, tc.callsign, instruction=tc.instruction)
            if is_meas:
                for k, v in ct.items():
                    compiled_times.setdefault(k, []).append(v)

        # Generic rules (LLM) - single measurement (too expensive for many iterations)
        generic_results: Dict[str, EvalResult] = {}
        if tc.eval_type == "generic":
            for rule_id, desc in rule_descriptions.items():
                latency_ms, result = benchmark.benchmark_generic(tc, rule_id, desc)
                generic_times.setdefault(rule_id, []).append(latency_ms)
                generic_results[rule_id] = result

        # Full pipeline - single measurement
        pipe_ms, pipe_steps, alerts = benchmark.benchmark_pipeline(tc)

        # Build benchmark result
        tc_bench = TcBenchmark(
            test_case_id=tc.id,
            bert_ms=sum(bert_times) / len(bert_times) if bert_times else 0.0,
            native_rules={k: sum(v) / len(v) for k, v in native_times.items()},
            compiled_rules={k: sum(v) / len(v) for k, v in compiled_times.items()},
            generic_rules={k: sum(v) / len(v) for k, v in generic_times.items()},
            pipeline_e2e_ms=pipe_ms,
            pipeline_steps=pipe_steps or {},
        )

        # Collect eval results (from compiled + generic)
        eval_results = {}
        parsed_instruction = benchmark._parse_instruction(tc.instruction)
        for rule_id in benchmark.compiled_rules:
            ct_vals = compiled_times.get(rule_id, [])
            if ct_vals:
                avg = sum(ct_vals) / len(ct_vals)
                fn = benchmark.compiled_rules[rule_id]
                try:
                    ts = benchmark_runner.build_traffic_state(tc.traffic_state)
                    res = fn(ts, callsign=tc.callsign, instruction=parsed_instruction)
                    eval_results[rule_id] = EvalResult(
                        rule_id=rule_id,
                        satisfied=res.get("satisfied", True),
                        severity=res.get("severity", "INFO"),
                        latency_ms=avg,
                        explanation=res.get("explanation", ""),
                    )
                except Exception as e:
                    eval_results[rule_id] = EvalResult(
                        rule_id=rule_id, satisfied=True, severity="INFO",
                        latency_ms=avg, error=str(e),
                    )

        eval_results.update(generic_results)
        tc_bench.eval_results = eval_results

        # Print summary line
        parts = [f"BERT={tc_bench.bert_ms:.0f}ms" if tc_bench.bert_ms > 0 else "BERT=skip"]
        if native_times:
            avg_n = sum(sum(v) for v in native_times.values()) / max(len(native_times), 1)
            parts.append(f"Native={avg_n:.1f}ms")
        if compiled_times:
            avg_c = sum(sum(v) for v in compiled_times.values()) / max(len(compiled_times), 1)
            parts.append(f"Compiled={avg_c:.1f}ms")
        if generic_times:
            avg_g = sum(sum(v) for v in generic_times.values()) / max(len(generic_times), 1)
            parts.append(f"GenericLLM={avg_g:.0f}ms")
        parts.append(f"Pipeline={pipe_ms:.1f}ms")
        print(f"    {' | '.join(parts)}")

        all_benchmarks.append(tc_bench)

        # Save partial results after each test case
        partial[tc.id] = tc_bench
        save_partial(cfg.results_dir, partial)

    print()

    # Aggregate
    results = E6Results(config=cfg, all_benchmarks=all_benchmarks)
    results.latency = aggregate_latency(all_benchmarks)
    results.accuracy = aggregate_accuracy(all_benchmarks, test_cases)

    # Judge
    if judge:
        results.judge_scores = run_judge_all(judge, test_cases, all_benchmarks)

    # Pipeline steps
    for bench in all_benchmarks:
        if bench.pipeline_steps:
            results.pipeline_steps[bench.test_case_id] = bench.pipeline_steps

    # Ranking by latency (lower is better)
    latency_scores = {}
    for sname in ["bert", "native", "compiled", "generic", "pipeline"]:
        ls = results.latency.get(sname)
        if ls and ls.avg_ms > 0:
            latency_scores[sname] = ls.avg_ms

    results.ranking = [
        {
            "component": k,
            "avg_latency_ms": round(v, 2),
        }
        for k, v in sorted(latency_scores.items(), key=lambda x: x[1])
    ]

    return results
