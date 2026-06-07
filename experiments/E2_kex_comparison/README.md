# E2: KEX Comparison Between Models and Ground Truth

## Overview

Evaluates how well different LLM models perform on the task of **knowledge extraction** from aeronautical documents. Compares model-generated extractions (entities, relationships, events, rules, procedures) against a manually curated ground truth using structural, content, cross-reference, and semantic metrics.

## Directory Structure

```
E2_kex_comparison/
├── ground_truth/                      # Your manually corrected pseudo-GT
│   ├── pagina_1.json
│   ├── pagina_2.json
│   └── ...
│
├── models/                            # Model output directories (same format as KEX output)
│   ├── ICAO Standard Phraseology(gemma4:31b-cloud)/
│   │   ├── pagina_1.json
│   │   └── ...
│   └── ICAO Standard Phraseology(gpt-oss:120b-cloud)/
│       └── ...
│
├── src/
│   ├── config.py            # Configuration (paths, judge config, metric weights)
│   ├── loader.py            # Load KEX outputs + errors
│   ├── matcher.py           # Fuzzy matching GT ↔ model
│   ├── metrics.py           # Structural + content + cross-ref metrics
│   ├── llm_judge.py         # LLM-as-a-judge with Instructor
│   ├── dedup_prompts.py     # Batch LLM prompts for semantic dedup detection
│   ├── dedup.py             # Cross-page semantic dedup (UnionFind, clustering, report)
│   ├── evaluator.py         # Orchestrate evaluation
│   ├── report.py            # Generate JSON results + PNG figures
│   └── run.py               # CLI entry point
│
├── scripts/
│   └── dedup_model.py       # Standalone dedup tool for pseudo-GT creation
│
├── results/
│   ├── page_metrics.json             # Per-page detailed metrics
│   ├── summary.json                   # Ranking + aggregated scores
│   └── figures/
│       ├── precision_recall_f1.png
│       ├── type_f1_comparison.png
│       ├── semantic_scores_boxplot.png
│       ├── overall_score_comparison.png
│       ├── radar_comparison.png
│       ├── error_rate_heatmap.png
│       ├── per_page_f1.png
│       └── per_page_semantic.png
│
└── README.md
```

## Usage

```bash
# Using project venv
/path/to/.venv/bin/python src/run.py

# Skip LLM judge (faster, structural only)
/path/to/.venv/bin/python src/run.py --no-judge

# Custom LLM judge
/path/to/.venv/bin/python src/run.py \
    --judge-model gpt-4o \
    --judge-provider openai \
    --judge-api-key $OPENAI_API_KEY

# Custom weights
/path/to/.venv/bin/python src/run.py --semantic-weight 0.75

# Disable dedup detection (enabled by default)
/path/to/.venv/bin/python src/run.py --no-dedup

# Standalone: analyze duplicates in a model output (pseudo-GT debugging)
/path/to/.venv/bin/python scripts/dedup_model.py \
    --model-dir models/"ICAO Standard Phraseology(gemma4:31b-cloud)" \
    --threshold 0.80 --batch-size 10
```

## Metrics

### Structural (15%)
| Metric | Description |
|--------|-------------|
| **Precision/Recall/F1** | Bipartite fuzzy matching between GT and model items |
| **Count Accuracy** | 1 - \|predicted - GT\| / GT |

### Content (10%)
| Metric | Description |
|--------|-------------|
| **Exact Field Match** | % of fields that match exactly |
| **Enum Match** | % of enum fields correct (rule_type, modality, phase, etc.) |

### Cross-Reference Validity (15%)
| Metric | Description |
|--------|-------------|
| **Validity Ratio** | % of internal references that point to existing items |
| **Broken Refs** | List of invalid references |

### Semantic (40%)
| Metric | Description |
|--------|-------------|
| **LLM Judge Score** | 0-1 semantic equivalence score per chunk (holistic set comparison) |
| **Explanation** | LLM explanation for the chunk-level semantic evaluation |

### Dedup Detection (Diagnostic)
| Metric | Description |
|--------|-------------|
| **Duplicate Clusters** | Groups of semantically identical items within a model |
| **Dedup Rate** | % of items that are duplicates (redundant / total) |
| **Cluster Sizes** | Distribution of cluster sizes (2, 3, ... items per cluster) |
| **Type Breakdown** | Duplicates per type (entity, relation, event, rule, procedure) |

Detection uses **LLM-as-a-judge in batch mode** (1 call per reference vs up to N candidates, triangular). Short-circuit exact match after normalization (lower+strip) triggers at 1.0 without LLM call. Default threshold: ≥0.80. Union-Find merges transitive clusters (A≈B and A≈C → {A,B,C}).

### Overall Score
Weighted combination:
- Structural F1: 15%
- Content Match: 10%
- Cross-Ref Validity: 15%
- LLM Semantic: 60%

## Matching Strategy

Items are matched using **fuzzy text similarity** on the primary field:
- **Entity** → `text`
- **Relationship** → `subject_text + predicate + object_text`
- **Event** → `trigger_text`
- **Rule** → `trigger.description`
- **Procedure** → `name`

Threshold: 70% similarity → match (configurable via `--fuzzy-threshold`)

## LLM-as-a-Judge

- Uses `common/llm_client_factory.py` (Instructor)
- **Schema Pydantic**: `Judgment(similarity_score, explanation, matched_fields, unmatched_fields)` (pairwise)
- **Schema Pydantic**: `BatchJudgment(result: list[CandidateJudgment])` (batch)
- **Blind**: model name is never revealed to the judge
- **One call per matched pair** for maximum precision (evaluation)
- **One call per reference vs N candidates** for dedup (batch, triangular matrix)
- **Type-specific prompts** for each knowledge type

## Error Analysis

Automatically reads `pagina_N_errors.json` files and includes:
- Error count per page and model
- Error rate (errors / total items)
- Item count difference heatmap

## Output

- `page_metrics.json`: Every page, every model, every metric, every type
- `summary.json`: Model ranking by overall score
- `dedup_{model}.json`: Per-model duplicate cluster report (when dedup enabled)
- `figures/`: 8 visualization PNGs
