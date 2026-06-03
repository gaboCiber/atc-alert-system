# E1: Chunk Comparison Between Models and Ground Truth

## Overview

Evaluates how well different LLM models perform on the task of **logical chunk segmentation** of aeronautical documents (ICAO Phraseology). Compares model-generated chunks against a manually curated ground truth.

## Directory Structure

```
E1_chunk_comparison/
├── ground_truth/                      # Your manually corrected pseudo-GT
│   ├── pagina_1_chunks.json
│   ├── pagina_2_chunks.json
│   └── ...
│
├── models/                            # Model output directories (same format as KEX output)
│   ├── ICAO Standard Phraseology(gemma4:e4b)/
│   │   ├── pagina_1_chunks.json
│   │   ├── pagina_2_chunks.json
│   │   └── ...
│   └── ICAO Standard Phraseology(gpt-oss:20b)/
│       ├── pagina_1_chunks.json
│       └── ...
│
├── src/
│   ├── config.py      # Configuration (paths, metric weights)
│   ├── loader.py      # Load models + GT from directories
│   ├── metrics.py     # Structural + content metrics
│   ├── evaluator.py   # Run full evaluation
│   ├── report.py      # Generate JSON results + PNG figures
│   └── run.py         # CLI entry point
│
├── results/
│   ├── page_metrics.json             # Per-page detailed metrics
│   ├── summary.json                   # Ranking + aggregated scores
│   └── figures/
│       ├── chunk_count_distribution.png
│       ├── boundary_f1_per_page.png
│       ├── chunk_count_error_heatmap.png
│       ├── overall_score_comparison.png
│       ├── radar_comparison.png
│       ├── content_metrics_boxplot.png
│       ├── boundary_avg_error.png
│       └── page_by_page.png
│
└── README.md
```

## Chunks JSON Format

Each `pagina_N_chunks.json` follows this structure:

```json
{
  "page_number": 1,
  "granularity": "chunk",
  "total_chunks": 4,
  "chunks": [
    {"chunk_index": 0, "text": "...", "char_count": 58},
    {"chunk_index": 1, "text": "...", "char_count": 170},
    ...
  ]
}
```

## Usage

```bash
# Using project venv
python src/run.py

# With custom paths
python src/run.py \
    --gt-dir /path/to/ground_truth \
    --models-dir /path/to/models \
    --output /path/to/output

# Skip figures (faster)
python src/run.py --no-figures
```

## Metrics

### Structural
| Metric | Description |
|--------|-------------|
| **Chunk Count Error** | \|predicted - GT\| |
| **Chunk Count Accuracy** | 1 - (error / GT count) |
| **Boundary F1** | Bipartite matching F1 on chunk boundaries |
| **Boundary Precision/Recall** | From bipartite matching |
| **Boundary Avg Error** | Average index offset from GT boundaries |

### Content
| Metric | Description |
|--------|-------------|
| **Matched Content F1** | F1 on per-matched-pair chunk similarity (via bipartite matching). Each matched pair is scored by `fuzz.ratio`, unmatched chunks score 0. |
| **Boundary Integrity** | Proportion of chunk boundaries that align with NLTK sentence boundaries. Measures segmentation quality independent of GT. |

### Overall Score
Weighed combination:
- Chunk Count Accuracy: 20%
- Boundary F1: 30%
- Matched Content F1: 30%
- Boundary Integrity: 20%

## Output

- `page_metrics.json`: Every page, every model, every metric
- `summary.json`: Model ranking by overall score with all component metrics
- `figures/`: 8 visualization PNGs (distribution, heatmaps, radar, boxplots)

## Creating Ground Truth

1. Run KEX extraction with your best model candidate
2. Copy the `pagina_N_chunks.json` files to `ground_truth/`
3. Manually correct boundaries and content
4. Re-run: the GT stays fixed while models vary

## Notes

- All model subdirectories under `models/` are auto-discovered
- Only pages present in **both** GT and at least one model are evaluated
- Supports any number of models and pages