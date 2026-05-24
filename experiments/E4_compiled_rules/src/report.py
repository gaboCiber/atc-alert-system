import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import E4Config
from evaluator import EvaluationResults, ModelSummaryMetrics


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _short_name(name: str) -> str:
    return name.split("(")[0].strip().replace(" ", "_")[:20]


def plot_overall_score_comparison(
    results: EvaluationResults,
    cfg: E4Config,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(8, len(results.model_names) * 1.5), 6))

    models = [_short_name(m) for m in results.model_names]
    scores = [results.summaries[m].overall_score for m in results.model_names]

    sorted_indices = np.argsort(scores)[::-1]
    models = [models[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    bars = ax.barh(models, scores, color=colors, edgecolor="black", linewidth=0.5)

    for bar, score in zip(bars, scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center", fontsize=10)

    ax.set_xlabel("Overall Score", fontsize=11)
    ax.set_title("Model Ranking by Overall Score", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(scores) * 1.15)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_metric_breakdown(
    results: EvaluationResults,
    cfg: E4Config,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(10, len(results.model_names) * 1.8), 6))

    metrics = ["classification", "validation", "execution", "semantic"]
    metric_labels = ["Classification\nAccuracy", "Validation\nPass Rate", "Execution\nMatch Rate", "Semantic\nScore"]
    n_metrics = len(metrics)

    x = np.arange(len(results.model_names))
    width = 0.8 / n_metrics

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = []
        for model in results.model_names:
            sm = results.summaries[model]
            if metric == "classification":
                v = sm.classification_metrics.accuracy if sm.classification_metrics else 0.0
            elif metric == "validation":
                v = sm.validation_pass_rate
            elif metric == "execution":
                v = sm.execution_metrics.match_rate if sm.execution_metrics else 0.0
            else:
                v = sm.semantic_metrics.mean_score if sm.semantic_metrics else 0.0
            values.append(v)

        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label, alpha=0.8)

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Metric Breakdown by Model", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_name(m) for m in results.model_names], rotation=45, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    return fig


def plot_validation_breakdown(
    results: EvaluationResults,
    cfg: E4Config,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(10, len(results.model_names) * 1.5), 6))

    models = [_short_name(m) for m in results.model_names]
    pass_rates = []
    total_rules = []

    for model in results.model_names:
        sm = results.summaries[model]
        pass_rates.append(sm.validation_pass_rate)
        total_rules.append(sm.validation_total)

    x = np.arange(len(models))
    bars = ax.bar(x, pass_rates, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(models))), edgecolor="black", linewidth=0.5)

    for bar, total in zip(bars, total_rules):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"n={total}", ha="center", fontsize=9)

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Validation Pass Rate", fontsize=11)
    ax.set_title("Code Validation Pass Rate", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    return fig


def plot_radar_comparison(
    results: EvaluationResults,
    cfg: E4Config,
) -> plt.Figure:
    metrics_names = ["Classification\nAccuracy", "Validation\nPass Rate", "Execution\nMatch Rate", "Semantic\nScore"]
    n_metrics = len(metrics_names)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(results.model_names)))
    for i, model in enumerate(results.model_names):
        sm = results.summaries[model]
        values = [
            sm.classification_metrics.accuracy if sm.classification_metrics else 0.0,
            sm.validation_pass_rate,
            sm.execution_metrics.match_rate if sm.execution_metrics else 0.0,
            sm.semantic_metrics.mean_score if sm.semantic_metrics else 0.0,
        ]
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, label=_short_name(model), color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Radar", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    return fig


def plot_execution_match_rate(
    results: EvaluationResults,
    cfg: E4Config,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(8, len(results.model_names) * 1.5), 6))

    models = [_short_name(m) for m in results.model_names]
    match_rates = []
    for model in results.model_names:
        em = results.summaries[model].execution_metrics
        match_rates.append(em.match_rate if em else 0.0)

    sorted_indices = np.argsort(match_rates)[::-1]
    models = [models[i] for i in sorted_indices]
    match_rates = [match_rates[i] for i in sorted_indices]

    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(models)))
    bars = ax.barh(models, match_rates, color=colors, edgecolor="black", linewidth=0.5)

    for bar, mr in zip(bars, match_rates):
        ax.text(mr + 0.01, bar.get_y() + bar.get_height() / 2, f"{mr:.2%}", va="center", fontsize=10)

    ax.set_xlabel("Execution Match Rate", fontsize=11)
    ax.set_title("Code Execution Match Rate vs Ground Truth", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(match_rates) * 1.15)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return fig


def generate_report(
    results: EvaluationResults,
    cfg: E4Config,
    output_dir: Optional[Path] = None,
    save_figures: bool = True,
) -> dict:
    if output_dir is None:
        output_dir = cfg.results_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "overall_score_comparison": plot_overall_score_comparison(results, cfg),
        "metric_breakdown": plot_metric_breakdown(results, cfg),
        "validation_breakdown": plot_validation_breakdown(results, cfg),
        "radar_comparison": plot_radar_comparison(results, cfg),
        "execution_match_rate": plot_execution_match_rate(results, cfg),
    }

    if save_figures:
        for name, fig in figures.items():
            fig.savefig(cfg.figures_dir / f"{name}.png", bbox_inches="tight", dpi=150)
            plt.close(fig)

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    summary_path = output_dir / "summary.json"
    sorted_models = sorted(results.model_names, key=lambda m: results.summaries[m].overall_score, reverse=True)
    summary_data = {
        "ranking": [
            {
                "rank": i + 1,
                "model": m,
                "overall_score": results.summaries[m].overall_score,
                "classification_accuracy": results.summaries[m].classification_metrics.accuracy if results.summaries[m].classification_metrics else 0.0,
                "validation_pass_rate": results.summaries[m].validation_pass_rate,
                "execution_match_rate": results.summaries[m].execution_metrics.match_rate if results.summaries[m].execution_metrics else 0.0,
                "semantic_score": results.summaries[m].semantic_metrics.mean_score if results.summaries[m].semantic_metrics else 0.0,
            }
            for i, m in enumerate(sorted_models)
        ]
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    report = {
        "results": str(results_path),
        "summary": str(summary_path),
        "figures": {name: str(cfg.figures_dir / f"{name}.png") for name in figures},
        "ranking": summary_data["ranking"],
    }

    return report