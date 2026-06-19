#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

import json
import matplotlib.pyplot as plt
import numpy as np

from config import E4Config


@dataclass
class ClassificationMetrics:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    total_rules: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0


@dataclass
class ExecutionMetrics:
    match_rate: float = 0.0
    successful_executions: int = 0
    failed_executions: int = 0


@dataclass
class SemanticMetrics:
    mean_score: float = 0.0
    count: int = 0


@dataclass
class ModelSummaryMetrics:
    model: str = ""
    classification_metrics: Optional[ClassificationMetrics] = None
    validation_pass_rate: float = 0.0
    validation_total: int = 0
    execution_metrics: Optional[ExecutionMetrics] = None
    semantic_metrics: Optional[SemanticMetrics] = None
    overall_score: float = 0.0


@dataclass
class EvaluationResults:
    model_names: list = field(default_factory=list)
    summaries: Dict[str, ModelSummaryMetrics] = field(default_factory=dict)


def _short_name(name: str) -> str:
    return name.split("(")[0].strip().replace(" ", "_")[:20]


def plot_comparison_table(
    results: EvaluationResults,
    cfg: E4Config,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(10, len(results.model_names) * 2), 4))
    ax.axis("off")

    columns = ["Overall", "Classification", "Validation", "Execution", "Semantic"]

    rows = []
    for model in results.model_names:
        sm = results.summaries[model]
        rows.append([
            _short_name(model),
            f"{sm.overall_score:.1%}",
            f"{sm.classification_metrics.accuracy:.1%}" if sm.classification_metrics else "N/A",
            f"{sm.validation_pass_rate:.1%}",
            f"{sm.execution_metrics.match_rate:.1%}" if sm.execution_metrics else "N/A",
            f"{sm.semantic_metrics.mean_score:.1%}" if sm.semantic_metrics else "N/A",
        ])

    col_headers = ["Model"] + columns
    table = ax.table(cellText=rows, colLabels=col_headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    best_values = {}
    for col_idx in range(1, len(col_headers)):
        vals = []
        for r in rows:
            v = r[col_idx]
            if v != "N/A":
                vals.append(float(v.rstrip('%')) / 100)
        if vals:
            best_values[col_idx] = max(vals)

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#ecf0f1" if i % 2 == 0 else "#ffffff")
            if j in best_values:
                cell_text = cell.get_text().get_text()
                if cell_text != "N/A":
                    cell_val = float(cell_text.rstrip('%')) / 100
                    if abs(cell_val - best_values[j]) < 1e-6:
                        cell.set_text_props(fontweight="bold")

    ax.set_title("Model Comparison Summary", fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    return fig


def load_results_from_summary(summary_path: Path) -> EvaluationResults:
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ranking = data.get("ranking", [])
    model_names = [entry["model"] for entry in ranking]

    summaries = {}
    for entry in ranking:
        model = entry["model"]
        summaries[model] = ModelSummaryMetrics(
            model=model,
            overall_score=entry.get("overall_score", 0.0),
            classification_metrics=ClassificationMetrics(
                accuracy=entry.get("classification_accuracy", 0.0),
            ),
            validation_pass_rate=entry.get("validation_pass_rate", 0.0),
            execution_metrics=ExecutionMetrics(
                match_rate=entry.get("execution_match_rate", 0.0),
            ),
            semantic_metrics=SemanticMetrics(
                mean_score=entry.get("semantic_score", 0.0),
            ),
        )

    return EvaluationResults(model_names=model_names, summaries=summaries)


def main():
    parser = argparse.ArgumentParser(description="Generate comparison table from E4 results")
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Path to summary.json (default: results/summary.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the PNG (default: results/figures/comparison_table.png)",
    )
    args = parser.parse_args()

    cfg = E4Config()

    summary_path = Path(args.results) if args.results else cfg.results_dir / "summary.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found")
        sys.exit(1)

    print(f"Loading results from {summary_path}...")
    results = load_results_from_summary(summary_path)
    print(f"  Found {len(results.model_names)} models: {', '.join(results.model_names)}")

    fig = plot_comparison_table(results, cfg)

    output_dir = cfg.figures_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else output_dir / "comparison_table.png"

    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
