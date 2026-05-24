# E6: System-Level Latency & Accuracy Benchmark

## Overview

Measures end-to-end latency and alert accuracy of the complete ATC Alert System across three rule evaluation paths (**native**, **compiled**, **generic LLM**) on a per-instruction basis. Each instruction passes through BERT parsing, native/compiled/generic rule evaluation (depending on `eval_type`), and the full `AlertPipeline`.

## Directory Structure

```
E6_system_benchmark/
├── ground_truth/
│   └── test_cases.json          # 8 test cases with traffic state + expected alerts
│
├── src/
│   ├── config.py                # E6Config, LLMConfig, MetricConfig
│   ├── test_case_loader.py      # TestCase + ExpectedAlert dataclasses, JSON loader
│   ├── benchmark_runner.py      # SystemBenchmark (times BERT, native, compiled, generic, pipeline)
│   ├── metrics.py               # LatencyStats, AccuracyMetrics, aggregation
│   ├── semantic_judge.py        # LLM judge for response quality score
│   ├── evaluator.py             # Orchestrator: warmup + measure iterations
│   ├── report.py                # 6 figures + JSON summary
│   └── run.py                   # CLI entry point
│
├── results/
│   ├── summary.json             # Latency stats (avg/p50/p95/p99) + accuracy (P/R/F1) + judge scores
│   └── figures/
│       ├── latency_flame.png      # Avg latency per component (bar)
│       ├── latency_breakdown.png  # Avg vs P95 per individual rule
│       ├── latency_per_tc.png     # Latency per test case, grouped by component
│       ├── accuracy.png           # Precision/Recall/F1 per rule_id
│       ├── pipeline_steps.png     # Stacked bar of pipeline internal steps
│       └── judge_scores.png       # Semantic judge score per test case
│
└── README.md
```

## Test Cases

| ID | Instruction | Path | Expected |
|---|---|---|---|
| TC001 | "descend AMX123 to 4000 feet" | native+compiled | RULE001 violation (alt below MSA) |
| TC002 | "climb AMX123 to flight level 380" | native+compiled | No violations |
| TC003 | "turn LAP456 left heading 270" | native+compiled | RULE002 violation (separation loss) |
| TC004 | "cleared for takeoff runway 27" | native+compiled | RULE004 violation (occupied runway) |
| TC005 | "contact Barcelona approach one one eight decimal one" | generic | No violations (phraseology check) |
| TC006 | "reduce speed to two five zero knots" | native+compiled | No violations |
| TC007 | "AMX123 squawk seven five zero zero" | generic | No violations (comms quality) |
| TC008 | "cleared for takeoff runway 09" | native+compiled | No violations (clear runway) |

## What Gets Measured

Each test case runs through:

1. **BERT parser** (`Jzuluaga/bert-base-ner-atc-en-atco2-1h`) — entity extraction latency
2. **Native rules** — `RuleEngine.evaluate()` for ALTITUDE, SEPARATION, RUNWAY
3. **Compiled rules** — `evaluate(ts, callsign)` from E4-generated `.py` files
4. **Generic LLM** — Instructor client with `MD_JSON` mode evaluating the rule via prompt
5. **Full pipeline** — `AlertPipeline.process_instruction()` with all 8 internal steps

Native/compiled paths run multiple measure iterations (default: 15 after 2 warmup). Generic LLM runs once per test case (too expensive for many iterations).

## Requirements

- **Compiled rules**: `E4_compiled_rules/models/model_A(gemma4:e4b)/` (configurable with `--compiled-rules-dir`)
- **LLM model**: `gemma4:31b-cloud` via Ollama (default, configurable with `--model`)
- **BERT model**: `Jzuluaga/bert-base-ner-atc-en-atco2-1h` (auto-downloaded from HuggingFace)
- **Python deps**: `instructor`, `pydantic`, `matplotlib`, `numpy`, `transformers`, `torch`

## Usage

```bash
# Full run (default: gemma4:31b-cloud via Ollama)
uv run python experiments/E6_system_benchmark/src/run.py

# Skip LLM judge (faster, lower cost)
uv run python experiments/E6_system_benchmark/src/run.py --no-judge

# Custom LLM
uv run python experiments/E6_system_benchmark/src/run.py \
    --model llama3.2 \
    --base-url http://localhost:11434/v1

# Control iterations
uv run python experiments/E6_system_benchmark/src/run.py \
    --warmup 3 --measure 20
```

## Pipeline Timing Notes

Test cases without a detectable callsign in the instruction text (e.g. "contact Barcelona approach...", "reduce speed to two five zero knots") fail projection early — `StateProjector.create_projection()` returns `is_valid_projection=False` — causing the pipeline to return after step 3 (~0.2ms). Test cases with a callsign but unrecognized instruction types (e.g. squawk) run through all 8 steps but execute quickly because no violation logic triggers (~0.9ms). This is **by design**, not an error.

To see pipeline errors when they occur, set `verbose=True` in `benchmark_runner.py:SystemBenchmark.benchmark_pipeline()`.

## Metrics

### Latency

Per component and per rule: **min**, **avg**, **p50**, **p95**, **p99** (ms).

Components: `bert`, `native` (aggregate), `compiled` (aggregate), `generic` (aggregate), `pipeline`.

Individual rules tracked as `native_ALTITUDE`, `compiled_RULE001`, `generic_RULE003`, etc.

### Accuracy

Per rule: **Precision**, **Recall**, **F1**, **Severity Accuracy**.

- `satisfied = False` → alert expected/triggered (violation)
- `satisfied = True` → safe (no alert)

| | Predicted Violation | Predicted Safe |
|---|---|---|
| **Expected Violation** | True Positive | False Negative |
| **Expected Safe** | False Positive | True Negative |

### Semantic Judge Score

LLM evaluates each test case holistically (0.0–1.0) considering recall, precision, severity correctness, and response quality.
