import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import E2Config
from .evaluator import EvaluationResults, ModelSummaryMetrics, ModelPageMetrics
from .matcher import KEX_TYPES


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


TYPE_LABELS = {
    "entities": "Entities",
    "relationships": "Relationships",
    "events": "Events",
    "rules": "Rules",
    "procedures": "Procedures",
}


def plot_precision_recall_f1(
    results: EvaluationResults,
    cfg: E2Config,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["structural_precision", "structural_recall", "structural_f1"]
    labels = ["Precision", "Recall", "F1"]

    for ax, metric, label in zip(axes, metrics, labels):
        data_rows = []
        for model in results.model_names:
            sm = results.summaries[model]
            data_rows.append({
                "Model": _short_name(model),
                "Value": getattr(sm, metric),
            })
        df = pd.DataFrame(data_rows)
        sns.barplot(data=df, x="Model", y="Value", ax=ax, hue="Model", palette="tab10", legend=False)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_type_f1_comparison(
    results: EvaluationResults,
    cfg: E2Config,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(KEX_TYPES))
    width = 0.8 / len(results.model_names)

    for i, model in enumerate(results.model_names):
        sm = results.summaries[model]
        f1s = [sm.type_structural_f1.get(k, 0.0) for k in KEX_TYPES]
        offset = (i - len(results.model_names) / 2 + 0.5) * width
        ax.bar(x + offset, f1s, width, label=_short_name(model), alpha=0.8)

    ax.set_xlabel("Knowledge Type", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Structural F1 by Knowledge Type", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_LABELS[k] for k in KEX_TYPES])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    return fig


def plot_semantic_scores_boxplot(
    results: EvaluationResults,
    cfg: E2Config,
) -> plt.Figure:
    fig, axes = plt.subplots(1, len(KEX_TYPES), figsize=(20, 5))

    for ax, kex_type in zip(axes, KEX_TYPES):
        data_rows = []
        for model in results.model_names:
            for pm in results.page_results[model]:
                sem = pm.semantic_scores.get(kex_type)
                if sem and sem.scores:
                    for s in sem.scores:
                        data_rows.append({"Model": _short_name(model), "Score": s})

        df = pd.DataFrame(data_rows)
        if not df.empty:
            sns.boxplot(data=df, x="Model", y="Score", ax=ax, hue="Model", palette="tab10", legend=False)
        ax.set_title(TYPE_LABELS[kex_type], fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_overall_score_comparison(
    results: EvaluationResults,
    cfg: E2Config,
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
    cfg: E2Config,
) -> plt.Figure:
    metrics_names = [
        "Structural\nF1",
        "Content\nMatch",
        "Cross-Ref\nValidity",
        "Semantic\nScore",
        "Error\nScore",
    ]
    values_keys = ["structural_f1", "content_avg", "cross_ref_avg", "semantic_avg", "error_avg"]
    n_metrics = len(metrics_names)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(results.model_names)))
    for i, model in enumerate(results.model_names):
        sm = results.summaries[model]
        values = [getattr(sm, k) for k in values_keys]
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


def plot_error_rate_heatmap(
    results: EvaluationResults,
    cfg: E2Config,
) -> plt.Figure:
    from .loader import ExperimentData

    matrix = np.zeros((len(results.model_names), len(results.pages)))
    for i, model in enumerate(results.model_names):
        for j, page in enumerate(results.pages):
            pm_list = results.page_results[model]
            pm = next((p for p in pm_list if p.page == page), None)
            if pm:
                matrix[i, j] = pm.structural_metrics.error_rate

    labels = [_short_name(m) for m in results.model_names]
    page_labels = [f"P{p}" for p in results.pages]

    fig, ax = plt.subplots(figsize=(max(8, len(results.pages) * 1.2), max(5, len(results.model_names) * 1.2)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=page_labels,
        yticklabels=labels,
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Error Rate (errors / model items)"},
    )
    ax.set_xlabel("Page", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    ax.set_title("Extraction Error Rate per Page", fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_ranking_table(
    results: EvaluationResults,
    cfg: E2Config,
) -> plt.Figure:
    sorted_models = sorted(
        results.model_names, key=lambda m: results.summaries[m].overall_score, reverse=True
    )

    headers = ["Rank", "Model", "Overall", "Structural F1", "Content", "CrossRef", "Semantic", "Extraction\nQuality"]

    numeric_cols = list(range(2, len(headers)))
    higher_is_better = {2: True, 3: True, 4: True, 5: True, 6: True, 7: True}

    rows = []
    for rank, model in enumerate(sorted_models, 1):
        sm = results.summaries[model]
        rows.append([
            rank,
            _short_name(model),
            f"{sm.overall_score:.3f}",
            f"{sm.structural_f1:.3f}",
            f"{sm.content_avg:.3f}",
            f"{sm.cross_ref_avg:.3f}",
            f"{sm.semantic_avg:.3f}",
            f"{sm.error_avg:.3f}",
        ])

    best_vals = {}
    for col_idx in numeric_cols:
        vals = [float(row[col_idx]) for row in rows]
        if higher_is_better[col_idx]:
            best_vals[col_idx] = max(vals)
        else:
            best_vals[col_idx] = min(vals)

    n_rows = len(rows)
    row_height = 0.45
    fig_height = n_rows * row_height + 1.5
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    col_widths = [0.05, 0.17, 0.08, 0.09, 0.08, 0.08, 0.09, 0.08]
    for (row, col), cell in table.get_celld().items():
        if col < len(col_widths):
            cell.set_width(col_widths[col])

    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    for i in range(n_rows):
        model_name = sorted_models[i]
        is_gemini = "gemini" in model_name.lower()
        for j in range(len(headers)):
            cell = table[i + 1, j]
            if is_gemini:
                cell.set_facecolor("#fff3cd")
                cell.set_text_props(fontweight="bold")
            elif i % 2 == 0:
                cell.set_facecolor("#f8f9fa")
            else:
                cell.set_facecolor("#ffffff")
            if j == 1:
                cell.set_text_props(ha="left")
            if j in numeric_cols and not is_gemini:
                val = float(rows[i][j])
                if val == best_vals[j]:
                    cell.set_facecolor("#d4edda")
                    cell.set_text_props(fontweight="bold")

    for i in range(n_rows + 1):
        table[i, 0].set_text_props(ha="center")

    ax.set_title(
        "Model Ranking by Overall Score",
        fontsize=13, fontweight="bold", pad=10, y=0.88
    )

    fig.subplots_adjust(top=0.7)

    plt.tight_layout()
    return fig


def plot_cross_ref_heatmap(
    results: EvaluationResults,
    cfg: E2Config,
) -> plt.Figure:
    matrix = np.zeros((len(results.model_names), len(results.pages)))
    for i, model in enumerate(results.model_names):
        for j, page in enumerate(results.pages):
            pm = next((p for p in results.page_results[model] if p.page == page), None)
            if pm:
                matrix[i, j] = pm.structural_metrics.overall_cross_ref

    labels = [_short_name(m) for m in results.model_names]
    page_labels = [f"P{p}" for p in results.pages]

    fig, ax = plt.subplots(figsize=(max(8, len(results.pages) * 1.2), max(5, len(results.model_names) * 1.2)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        xticklabels=page_labels,
        yticklabels=labels,
        ax=ax,
        vmin=0,
        vmax=1,
        center=0.5,
        cbar_kws={"label": "Cross-Ref Validity (0 = broken, 1 = valid)"},
    )
    ax.set_xlabel("Page", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    ax.set_title("Cross-Reference Validity per Page", fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_type_cross_ref_comparison(
    results: EvaluationResults,
    cfg: E2Config,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(KEX_TYPES))
    width = 0.8 / len(results.model_names)

    for i, model in enumerate(results.model_names):
        sm = results.summaries[model]
        vals = [sm.type_cross_ref.get(k, 0.0) for k in KEX_TYPES]
        offset = (i - len(results.model_names) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=_short_name(model), alpha=0.8)

    ax.set_xlabel("Knowledge Type", fontsize=11)
    ax.set_ylabel("Cross-Ref Validity", fontsize=11)
    ax.set_title("Cross-Reference Validity by Knowledge Type", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_LABELS[k] for k in KEX_TYPES])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    return fig


def plot_per_page_f1(
    results: EvaluationResults,
    cfg: E2Config,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(results.pages))
    for model in results.model_names:
        short = _short_name(model)
        # Create a list of F1 scores for each page, filling missing pages with 0
        f1s = []
        for page in results.pages:
            # Find the ModelPageMetrics for this model and page
            pm_for_page = next((pm for pm in results.page_results[model] if pm.page == page), None)
            if pm_for_page:
                f1s.append(pm_for_page.structural_metrics.overall_f1)
            else:
                f1s.append(0.0)  # No results for this page/model combination
        ax.plot(x, f1s, "o-", label=short, linewidth=2, markersize=5)

    ax.set_xlabel("Page", fontsize=11)
    ax.set_ylabel("Overall F1", fontsize=11)
    ax.set_title("Overall F1 Score per Page", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in results.pages])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    return fig


def plot_per_page_semantic(
    results: EvaluationResults,
    cfg: E2Config,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(results.pages))
    for model in results.model_names:
        short = _short_name(model)
        # Create a list of semantic scores for each page, filling missing pages with 0
        sems = []
        for page in results.pages:
            # Find the ModelPageMetrics for this model and page
            pm_for_page = next((pm for pm in results.page_results[model] if pm.page == page), None)
            if pm_for_page:
                sems.append(pm_for_page.overall_semantic)
            else:
                sems.append(0.0)  # No results for this page/model combination
        ax.plot(x, sems, "s--", label=short, linewidth=2, markersize=5)

    ax.set_xlabel("Page", fontsize=11)
    ax.set_ylabel("Overall Semantic Score", fontsize=11)
    ax.set_title("Overall Semantic Score per Page", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in results.pages])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    return fig


def generate_report(
    results: EvaluationResults,
    cfg: E2Config,
    output_dir: Optional[Path] = None,
    save_figures: bool = True,
) -> dict:
    if output_dir is None:
        output_dir = cfg.results_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "precision_recall_f1": plot_precision_recall_f1(results, cfg),
        "type_f1_comparison": plot_type_f1_comparison(results, cfg),
        "semantic_scores_boxplot": plot_semantic_scores_boxplot(results, cfg),
        "overall_score_comparison": plot_overall_score_comparison(results, cfg),
        "radar_comparison": plot_radar_comparison(results, cfg),
        "error_rate_heatmap": plot_error_rate_heatmap(results, cfg),
        "per_page_f1": plot_per_page_f1(results, cfg),
        "per_page_semantic": plot_per_page_semantic(results, cfg),
        "ranking_table": plot_ranking_table(results, cfg),
        "cross_ref_heatmap": plot_cross_ref_heatmap(results, cfg),
        "type_cross_ref_comparison": plot_type_cross_ref_comparison(results, cfg),
    }

    if save_figures:
        for name, fig in figures.items():
            fig.savefig(cfg.figures_dir / f"{name}.png", bbox_inches="tight", dpi=150)
            plt.close(fig)

    page_metrics_path = output_dir / "page_metrics.json"
    with open(page_metrics_path, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    summary_path = output_dir / "summary.json"
    sorted_models = sorted(results.model_names, key=lambda m: results.summaries[m].overall_score, reverse=True)
    summary_data = {
        "ranking": [
            {
                "rank": i + 1,
                "model": m,
                "overall_score": results.summaries[m].overall_score,
                "structural_f1": results.summaries[m].structural_f1,
                "content_avg": results.summaries[m].content_avg,
                "cross_ref_avg": results.summaries[m].cross_ref_avg,
                "semantic_avg": results.summaries[m].semantic_avg,
                "dedup_avg": results.summaries[m].dedup_avg,
                "error_avg": results.summaries[m].error_avg,
            }
            for i, m in enumerate(sorted_models)
        ]
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    for model_name, report_obj in results.dedup_reports.items():
        dedup_path = output_dir / f"dedup_{model_name}.json"
        with open(dedup_path, "w", encoding="utf-8") as f:
            json.dump(report_obj.to_dict(), f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    report = {
        "page_metrics": str(page_metrics_path),
        "summary": str(summary_path),
        "figures": {name: str(cfg.figures_dir / f"{name}.png") for name in figures},
        "ranking": summary_data["ranking"],
    }

    return report
