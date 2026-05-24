import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import E3Config
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
    return name.replace("_", " ").replace("-", " ").title()[:25]


def plot_wer_comparison(
    results: EvaluationResults,
    cfg: E3Config,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(8, len(results.model_names) * 1.5), 6))

    models = [_short_name(m) for m in results.model_names]
    wers = [results.summaries[m].average_wer for m in results.model_names]

    sorted_indices = np.argsort(wers)
    models = [models[i] for i in sorted_indices]
    wers = [wers[i] for i in sorted_indices]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    bars = ax.barh(models, wers, color=colors, edgecolor="black", linewidth=0.5)

    for bar, wer in zip(bars, wers):
        ax.text(wer + 0.005, bar.get_y() + bar.get_height() / 2, f"{wer:.2%}", va="center", fontsize=10)

    ax.set_xlabel("Word Error Rate (WER)", fontsize=11)
    ax.set_title("Model Ranking by WER (lower is better)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(wers) * 1.15)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_error_breakdown(
    results: EvaluationResults,
    cfg: E3Config,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(8, len(results.model_names) * 1.5), 6))

    models = [_short_name(m) for m in results.model_names]
    subs = [results.summaries[m].total_substitutions for m in results.model_names]
    ins = [results.summaries[m].total_insertions for m in results.model_names]
    dels = [results.summaries[m].total_deletions for m in results.model_names]

    x = np.arange(len(models))
    width = 0.25

    ax.bar(x - width, subs, width, label="Substitutions", color="#e74c3c")
    ax.bar(x, ins, width, label="Insertions", color="#f39c12")
    ax.bar(x + width, dels, width, label="Deletions", color="#3498db")

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Error Count", fontsize=11)
    ax.set_title("Error Type Breakdown by Model", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_per_sample_wer_boxplot(
    results: EvaluationResults,
    cfg: E3Config,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(8, len(results.model_names) * 1.5), 6))

    data_rows = []
    for model in results.model_names:
        sm = results.summaries[model]
        for wer in sm.per_sample_wer:
            data_rows.append({"Model": _short_name(model), "WER": wer})

    df = pd.DataFrame(data_rows)
    sns.boxplot(data=df, x="Model", y="WER", ax=ax, hue="Model", palette="tab10", legend=False)

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Word Error Rate (WER)", fontsize=11)
    ax.set_title("Per-Sample WER Distribution", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_metrics_radar(
    results: EvaluationResults,
    cfg: E3Config,
) -> plt.Figure:
    metrics_names = ["WER\n(inverted)", "MER\n(inverted)", "WIP", "WIL\n(inverted)"]
    n_metrics = len(metrics_names)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(results.model_names)))
    for i, model in enumerate(results.model_names):
        sm = results.summaries[model]
        values = [
            1.0 - sm.average_wer,
            1.0 - sm.average_mer if sm.average_mer > 0 else 0.0,
            sm.average_wip if sm.average_wip > 0 else 0.0,
            1.0 - sm.average_wil if sm.average_wil > 0 else 0.0,
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


def plot_wer_vs_length(
    results: EvaluationResults,
    cfg: E3Config,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(results.model_names)))
    for i, model in enumerate(results.model_names):
        sm = results.summaries[model]
        ref_words = [r.num_ref_words for r in sm.results]
        wers = sm.per_sample_wer
        ax.scatter(ref_words, wers, alpha=0.6, label=_short_name(model), color=colors[i], edgecolors="black", linewidth=0.5)

    ax.set_xlabel("Reference Word Count", fontsize=11)
    ax.set_ylabel("Word Error Rate (WER)", fontsize=11)
    ax.set_title("WER vs Reference Length", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_model_comparison_table(
    results: EvaluationResults,
    cfg: E3Config,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(10, len(results.model_names) * 2), 4))
    ax.axis("off")

    rows = []
    for model in results.model_names:
        sm = results.summaries[model]
        rows.append([
            _short_name(model),
            f"{sm.average_wer:.2%}",
            f"{sm.average_mer:.2%}" if sm.average_mer > 0 else "N/A",
            f"{sm.average_wil:.2%}" if sm.average_wil > 0 else "N/A",
            f"{sm.average_wip:.2%}" if sm.average_wip > 0 else "N/A",
            str(sm.total_substitutions),
            str(sm.total_insertions),
            str(sm.total_deletions),
        ])

    columns = ["Model", "WER", "MER", "WIL", "WIP", "Subs", "Ins", "Dels"]
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#ecf0f1" if i % 2 == 0 else "#ffffff")

    ax.set_title("Model Comparison Summary", fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    return fig


def generate_report(
    results: EvaluationResults,
    cfg: E3Config,
    output_dir: Optional[Path] = None,
    save_figures: bool = True,
) -> dict:
    if output_dir is None:
        output_dir = cfg.results_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "wer_comparison": plot_wer_comparison(results, cfg),
        "error_breakdown": plot_error_breakdown(results, cfg),
        "per_sample_wer_boxplot": plot_per_sample_wer_boxplot(results, cfg),
        "metrics_radar": plot_metrics_radar(results, cfg),
        "wer_vs_length": plot_wer_vs_length(results, cfg),
        "comparison_table": plot_model_comparison_table(results, cfg),
    }

    if save_figures:
        for name, fig in figures.items():
            fig.savefig(cfg.figures_dir / f"{name}.png", bbox_inches="tight", dpi=150)
            plt.close(fig)

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    summary_path = output_dir / "summary.json"
    sorted_models = sorted(results.model_names, key=lambda m: results.summaries[m].average_wer)
    summary_data = {
        "ranking": [
            {
                "rank": i + 1,
                "model": m,
                "wer": results.summaries[m].average_wer,
                "mer": results.summaries[m].average_mer,
                "wil": results.summaries[m].average_wil,
                "wip": results.summaries[m].average_wip,
                "substitutions": results.summaries[m].total_substitutions,
                "insertions": results.summaries[m].total_insertions,
                "deletions": results.summaries[m].total_deletions,
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
