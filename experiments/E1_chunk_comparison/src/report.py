import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import E1Config, MetricConfig


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
from evaluator import EvaluationResults, ModelSummaryMetrics, PageMetrics
from metrics import StructuralMetrics, ContentMetrics


plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "sans-serif"


def _short_name(name: str) -> str:
    return name.split("(")[0].strip().replace(" ", "_")[:20]


def plot_chunk_count_distribution(
    results: EvaluationResults,
    cfg: E1Config,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))

    data_rows = []
    for model in results.model_names:
        for pm in results.page_results[model]:
            data_rows.append(
                {
                    "Page": pm.page,
                    "Model": _short_name(model),
                    "Predicted": pm.structural.chunk_count,
                    "Ground Truth": pm.structural.gt_chunk_count,
                }
            )
    df = pd.DataFrame(data_rows)

    x = np.arange(len(results.pages))
    width = 0.8 / len(results.model_names)

    for i, model in enumerate(results.model_names):
        short = _short_name(model)
        counts = [results.page_results[model][j].structural.chunk_count for j in range(len(results.pages))]
        offset = (i - len(results.model_names) / 2 + 0.5) * width
        ax.bar(x + offset, counts, width, label=short, alpha=0.8)

    gt_counts = [results.page_results[results.model_names[0]][j].structural.gt_chunk_count for j in range(len(results.pages))]
    ax.plot(x, gt_counts, "ko--", linewidth=2, markersize=6, label="Ground Truth", zorder=5)

    ax.set_xlabel("Page", fontsize=11)
    ax.set_ylabel("Chunk Count", fontsize=11)
    ax.set_title("Chunk Count per Page: Models vs Ground Truth", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in results.pages])
    ax.legend(fontsize=9, ncol=min(len(results.model_names) + 1, 5))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_boundary_f1_per_page(
    results: EvaluationResults,
    cfg: E1Config,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(results.pages))
    for model in results.model_names:
        short = _short_name(model)
        bf1 = [pm.structural.boundary_f1 for pm in results.page_results[model]]
        ax.plot(x, bf1, "o-", label=short, linewidth=2, markersize=5)

    ax.set_xlabel("Page", fontsize=11)
    ax.set_ylabel("Boundary F1", fontsize=11)
    ax.set_title("Boundary F1 Score per Page", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in results.pages])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    return fig


def plot_chunk_count_error_heatmap(
    results: EvaluationResults,
    cfg: E1Config,
) -> plt.Figure:
    matrix = np.zeros((len(results.model_names), len(results.pages)))
    for i, model in enumerate(results.model_names):
        for j, pm in enumerate(results.page_results[model]):
            matrix[i, j] = pm.structural.chunk_count_error

    labels = [_short_name(m) for m in results.model_names]
    page_labels = [f"P{p}" for p in results.pages]

    fig, ax = plt.subplots(figsize=(max(8, len(results.pages) * 1.2), max(5, len(results.model_names) * 1.2)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn_r",
        xticklabels=page_labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Chunk Count Error"},
        center=0,
    )
    ax.set_xlabel("Page", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    ax.set_title("Chunk Count Error Heatmap (0 = perfect)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_overall_score_comparison(
    results: EvaluationResults,
    cfg: E1Config,
) -> plt.Figure:
    models = [_short_name(m) for m in results.model_names]
    scores = [results.summaries[m].overall_score for m in results.model_names]

    sorted_indices = np.argsort(scores)[::-1]
    models = [models[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 6))
    bars = ax.barh(models, scores, color=colors, edgecolor="black", linewidth=0.5)

    for bar, score in zip(bars, scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center", fontsize=10)

    ax.set_xlabel("Overall Score", fontsize=11)
    ax.set_title("Model Ranking by Overall Score", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(scores) * 1.15)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_radar_comparison(
    results: EvaluationResults,
    cfg: E1Config,
) -> plt.Figure:
    metrics_names = [
        "Chunk Count\nAccuracy",
        "Boundary\nF1",
        "Char F1",
        "Word F1",
        "ROUGE-L",
        "Fuzzy\nMatch",
    ]
    n_metrics = len(metrics_names)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(results.model_names)))
    for i, model in enumerate(results.model_names):
        sm = results.summaries[model]
        values = [
            sm.chunk_count_accuracy_mean,
            sm.boundary_f1_mean,
            sm.char_f1_mean,
            sm.word_f1_mean,
            sm.rouge_l_mean,
            sm.fuzzy_match_mean,
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


def plot_content_metrics_boxplot(
    results: EvaluationResults,
    cfg: E1Config,
) -> plt.Figure:
    metric_names = ["char_f1", "word_f1", "rouge_l", "fuzzy_match_ratio"]
    metric_labels = ["Char F1", "Word F1", "ROUGE-L", "Fuzzy Match"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, mname, mlabel in zip(axes, metric_names, metric_labels):
        data_rows = []
        for model in results.model_names:
            for pm in results.page_results[model]:
                val = getattr(pm.content, mname)
                data_rows.append({"Model": _short_name(model), "Value": val})

        df = pd.DataFrame(data_rows)
        sns.boxplot(data=df, x="Model", y="Value", ax=ax, hue="Model", palette="tab10", legend=False)
        ax.set_title(mlabel, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_boundary_avg_error_per_page(
    results: EvaluationResults,
    cfg: E1Config,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(results.pages))
    for model in results.model_names:
        short = _short_name(model)
        errs = [pm.structural.boundary_avg_error for pm in results.page_results[model]]
        ax.plot(x, errs, "s--", label=short, linewidth=2, markersize=5)

    ax.set_xlabel("Page", fontsize=11)
    ax.set_ylabel("Boundary Avg Error (chunks)", fontsize=11)
    ax.set_title("Boundary Average Error per Page", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in results.pages])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_page_by_page_comparison(
    results: EvaluationResults,
    cfg: E1Config,
) -> plt.Figure:
    metrics = ["boundary_f1", "char_f1", "word_f1", "rouge_l"]
    metric_labels = ["Boundary F1", "Char F1", "Word F1", "ROUGE-L"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, mname, mlabel in zip(axes, metrics, metric_labels):
        x = np.arange(len(results.pages))
        for model in results.model_names:
            short = _short_name(model)
            if mname.startswith("boundary"):
                vals = [pm.structural.boundary_f1 for pm in results.page_results[model]]
            else:
                vals = [getattr(pm.content, mname) for pm in results.page_results[model]]
            ax.plot(x, vals, "o-", label=short, linewidth=2, markersize=4)

        ax.set_xlabel("Page", fontsize=10)
        ax.set_ylabel(mlabel, fontsize=10)
        ax.set_title(mlabel, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"P{p}" for p in results.pages])
        ax.legend(fontsize=8, ncol=min(len(results.model_names), 3))
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    return fig


def generate_report(
    results: EvaluationResults,
    cfg: E1Config,
    output_dir: Optional[Path] = None,
    save_figures: bool = True,
) -> dict:
    if output_dir is None:
        output_dir = cfg.results_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "chunk_count_distribution": plot_chunk_count_distribution(results, cfg),
        "boundary_f1_per_page": plot_boundary_f1_per_page(results, cfg),
        "chunk_count_error_heatmap": plot_chunk_count_error_heatmap(results, cfg),
        "overall_score_comparison": plot_overall_score_comparison(results, cfg),
        "radar_comparison": plot_radar_comparison(results, cfg),
        "content_metrics_boxplot": plot_content_metrics_boxplot(results, cfg),
        "boundary_avg_error": plot_boundary_avg_error_per_page(results, cfg),
        "page_by_page": plot_page_by_page_comparison(results, cfg),
    }

    if save_figures:
        for name, fig in figures.items():
            fig.savefig(cfg.figures_dir / f"{name}.png", bbox_inches="tight", dpi=150)
            plt.close(fig)

    page_metrics_path = output_dir / "page_metrics.json"
    with open(page_metrics_path, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    summary_path = output_dir / "summary.json"
    summary_data = {
        "ranking": [
            {
                "rank": i + 1,
                "model": results.model_names[i],
                "overall_score": results.summaries[results.model_names[i]].overall_score,
                "boundary_f1_mean": results.summaries[results.model_names[i]].boundary_f1_mean,
                "char_f1_mean": results.summaries[results.model_names[i]].char_f1_mean,
                "word_f1_mean": results.summaries[results.model_names[i]].word_f1_mean,
                "rouge_l_mean": results.summaries[results.model_names[i]].rouge_l_mean,
            }
            for i in np.argsort([results.summaries[m].overall_score for m in results.model_names])[::-1]
        ]
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    report = {
        "page_metrics": str(page_metrics_path),
        "summary": str(summary_path),
        "figures": {name: str(cfg.figures_dir / f"{name}.png") for name in figures},
        "ranking": summary_data["ranking"],
    }

    return report