# E4: Compiled Rules Generation Comparison

## Overview

Evaluates how well different LLM models compile KEX (Knowledge Extractor) rules into executable Python code for the ATC Alert System. Compares classification decisions, validation pass rates, execution correctness, and semantic equivalence against a ground truth.

## Directory Structure

```
E4_compiled_rules/
├── ground_truth/
│   ├── expected_classification.json  # {"rule_id": {"is_compilable": bool, "reason": "..."}}
│   ├── reference_code/               # Reference Python implementations
│   │   ├── RULE001.py
│   │   └── ...
│   └── test_traffic_states/          # TrafficState JSON for execution testing
│       ├── altitude_violation.json
│       ├── separation_conflict.json
│       └── ...
│
├── models/                           # Model compilation outputs
│   ├── model_A/
│   │   ├── manifest.json
│   │   ├── RULE001.py
│   │   └── ...
│   └── model_B/
│
├── src/
│   ├── config.py      # Configuration (paths, judge config, metric weights)
│   ├── loader.py      # Load manifests, reference code, test states
│   ├── classifier_evaluator.py   # Classification accuracy vs GT
│   ├── validator.py   # AST validation (syntax, imports, signature, return)
│   ├── executor.py    # Execute code vs test TrafficStates
│   ├── semantic_judge.py  # LLM judge for semantic equivalence
│   ├── evaluator.py   # Orchestrate full evaluation
│   ├── report.py      # JSON + PNG figures
│   └── run.py         # CLI entry point
│
├── results/
│   ├── results.json
│   ├── summary.json
│   └── figures/
│       ├── overall_score_comparison.png
│       ├── metric_breakdown.png
│       ├── validation_breakdown.png
│       ├── radar_comparison.png
│       └── execution_match_rate.png
│
└── README.md
```

## Ground Truth Structure

### expected_classification.json

```json
{
  "RULE001": {"is_compilable": true, "reason": "measurable altitude constraint"},
  "RULE002": {"is_compilable": true, "reason": "measurable separation threshold"},
  "RULE003": {"is_compilable": false, "reason": "requires human judgment"}
}
```

### reference_code/

One `.py` file per rule, containing the reference implementation:

```python
def evaluate(traffic_state, callsign=None):
    """Reference implementation for RULE001"""
    aircraft = traffic_state.get_aircraft(callsign) if callsign else None
    if not aircraft:
        return {"satisfied": True, "details": {}, "explanation": "No aircraft", "severity": "INFO"}
    # ... reference logic
    return {"satisfied": True, "details": {}, "explanation": "...", "severity": "INFO"}
```

### test_traffic_states/

JSON files with TrafficState data for execution testing:

```json
{
  "sector_id": "TEST",
  "msa": 5000,
  "aircrafts": {"AAL123": {"position": {"altitude": 4000}, "flight_phase": "APPROACH"}},
  "runways": {},
  "expected_outcome": {"satisfied": false, "severity": "CRITICAL"}
}
```

## Usage

```bash
.venv/bin/python src/run.py

# Skip LLM judge
.venv/bin/python src/run.py --no-judge

# Custom judge
.venv/bin/python src/run.py \
    --judge-model llama3.2 \
    --judge-provider ollama
```

## Metrics

| Category | Weight | Metric |
|----------|--------|--------|
| **Classification** | 15% | Accuracy (compilable vs not_compilable) |
| **Validation** | 15% | Pass rate (AST, imports, signature, return) |
| **Execution** | 30% | Match rate vs expected outputs |
| **Semantic** | 40% | LLM judge semantic equivalence score |

### Overall Score
```
score = 0.15 * classification_accuracy
      + 0.15 * validation_pass_rate
      + 0.30 * execution_match_rate
      + 0.40 * semantic_score
```

## Validation Checks

The validator checks:
1. **Syntax**: Valid Python AST
2. **Function name**: `def evaluate(` exists
3. **Signature**: First arg is `traffic_state`
4. **Forbidden imports**: Only `math`, `datetime` allowed
5. **Forbidden names**: No `exec`, `eval`, `open`, `__import__`, etc.
6. **Return structure**: Dict with `satisfied`, `details`, `explanation`, `severity`