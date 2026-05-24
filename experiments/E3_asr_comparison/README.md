# E3: ASR Transcription Comparison Between Models and Ground Truth

## Overview

Evaluates how well different ASR (Automatic Speech Recognition) models transcribe ATC audio communications. Compares model-generated transcriptions against a manually curated ground truth using standard ASR metrics (WER, MER, WIL, WIP) with ATC-aware text normalization.

## Directory Structure

```
E3_asr_comparison/
├── ground_truth/                      # GT data (CSV, JSON, or TXT)
│   └── ground_truth.csv               # Format: id,text
│
├── transcriptions/                    # Transcription CSVs (one or more)
│   └── transcriptions.csv             # Format: model_name,sample1,sample2,...
│
├── src/
│   ├── config.py      # Configuration (paths, evaluation settings)
│   ├── loader.py      # Loads GT and transcription CSVs
│   ├── evaluator.py   # Wraps ASREvaluator from ASR module
│   ├── report.py      # Generate JSON results + PNG figures
│   └── run.py         # CLI entry point
│
├── results/
│   ├── results.json                   # Full per-sample results
│   ├── summary.json                   # Model ranking
│   └── figures/
│       ├── wer_comparison.png         # Horizontal bar chart
│       ├── error_breakdown.png        # Subs/Ins/Dels grouped bars
│       ├── per_sample_wer_boxplot.png # WER distribution
│       ├── metrics_radar.png          # WER/MER/WIL/WIP radar
│       ├── wer_vs_length.png          # Scatter plot
│       └── comparison_table.png       # Summary table
│
└── README.md
```

## Input Formats

### Ground Truth

**CSV** (recommended):
```csv
id,text
sample_001,maintain flight level three five zero
sample_002,cleared for takeoff runway zero nine
```

**JSON**:
```json
{
  "sample_001": "maintain flight level three five zero",
  "sample_002": "cleared for takeoff runway zero nine"
}
```

**TXT** (one sample per line, auto-numbered):
```
maintain flight level three five zero
cleared for takeoff runway zero nine
```

### Transcriptions

**CSV** (one file, multiple models):
```csv
model_name,sample_001,sample_002
whisper-large,maintain flight level 350,cleared for takeoff runway 09
faster-whisper,maintain flight level three five zero,cleared takeoff runway zero nine
```

Multiple CSV files in `transcriptions/` are automatically merged.

## Usage

```bash
# Using project venv
.venv/bin/python experiments/E3_asr_comparison/src/run.py

# Custom paths
.venv/bin/python experiments/E3_asr_comparison/src/run.py \
    --gt-dir /path/to/gt \
    --transcriptions-dir /path/to/transcriptions

# Skip ATC normalization (compare raw text)
.venv/bin/python experiments/E3_asr_comparison/src/run.py --no-atc-normalizer
```

## Metrics

| Metric | Description |
|--------|-------------|
| **WER** | Word Error Rate (substitutions + insertions + deletions) / reference words |
| **MER** | Match Error Rate (errors) / (hits + errors) |
| **WIL** | Word Information Lost |
| **WIP** | Word Information Preserved |
| **Error Breakdown** | Substitutions, Insertions, Deletions, Hits |

### ATC Normalization

By default, the evaluator applies ATC-specific text normalization before comparison:
- Callsign expansion (e.g., "UAL123" → "united one two three")
- Number expansion (e.g., "350" → "three five zero")
- ATC terminology normalization
- Punctuation removal
- Lowercasing

This ensures fair comparison between models that may format output differently.

## Output

- `results.json`: Full per-sample results with all metrics
- `summary.json`: Model ranking by WER
- `figures/`: 6 visualization PNGs
