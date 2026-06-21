"""Generación de reportes y gráficas académicas para E7."""

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluator import E7Results


PIPELINE_STEPS = [
    "INPUT_PROCESSING",
    "NORMALIZATION",
    "STATE_PROJECTION",
    "RULE_EVALUATION",
    "ALERT_GENERATION",
    "ALERT_PRESENTATION",
    "ATCO_DECISION",
    "FINAL_STATE_UPDATE",
]


def generate_report(results: E7Results, cfg, save_figures: bool = True) -> Dict[str, Any]:
    results_dir = cfg.results_dir
    figures_dir = cfg.figures_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig_paths = {}
    if save_figures:
        fig_paths.update(_plot_pipeline_accuracy_boxplot(results, figures_dir))
        fig_paths.update(_plot_e2e_latency(results, figures_dir))
        fig_paths.update(_plot_pipeline_steps(results, figures_dir))
        fig_paths.update(_plot_judge_distribution(results, figures_dir))
        fig_paths.update(_plot_recall_precision_scatter(results, figures_dir))
        fig_paths.update(_plot_latency_flame(results, figures_dir))

    summary = {
        "latency": results.latency,
        "accuracy": {
            "aggregate": results.aggregate_accuracy,
            "per_test_case": results.per_tc_accuracy,
        },
        "judge_scores": results.judge_scores,
    }

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "results": str(results_dir),
        "summary": str(summary_path),
        "figures": fig_paths,
    }


def _plot_pipeline_accuracy_boxplot(results: E7Results, figures_dir: Path) -> Dict[str, str]:
    metrics = ["precision", "recall", "f1"]
    data = {m: [] for m in metrics}
    for tc_acc in results.per_tc_accuracy.values():
        for m in metrics:
            data[m].append(tc_acc.get(m, 0.0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([data[m] for m in metrics], labels=[m.capitalize() for m in metrics])
    ax.set_ylabel("Score")
    ax.set_title("Pipeline Accuracy per Test Case")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    path = figures_dir / "system_overall_accuracy.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"system_overall_accuracy": str(path)}


def _plot_e2e_latency(results: E7Results, figures_dir: Path) -> Dict[str, str]:
    values = [r.total_ms for r in results.all_results if r.total_ms > 0]
    if not values:
        return {}

    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_vals = sorted(values)
    p50 = sorted_vals[len(sorted_vals) // 2]
    p95 = sorted_vals[int(len(sorted_vals) * 0.95)]
    ax.hist(values, bins=min(20, len(values)), color="#4C72B0", edgecolor="white")
    ax.axvline(p50, color="orange", linestyle="--", label=f"P50={p50:.0f}ms")
    ax.axvline(p95, color="red", linestyle="--", label=f"P95={p95:.0f}ms")
    ax.set_xlabel("End-to-end latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Pipeline E2E Latency Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = figures_dir / "latency_percentiles_stress.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"latency_percentiles_stress": str(path)}


def _plot_pipeline_steps(results: E7Results, figures_dir: Path) -> Dict[str, str]:
    step_avgs: Dict[str, List[float]] = {s: [] for s in PIPELINE_STEPS}
    for res in results.all_results:
        for step in res.steps.values():
            name = step.get("name", "")
            ms = step.get("execution_time_ms", 0.0)
            if name in step_avgs and ms > 0:
                step_avgs[name].append(ms)

    labels = []
    avgs = []
    p95s = []
    for name in PIPELINE_STEPS:
        vals = step_avgs[name]
        if not vals:
            continue
        labels.append(name.replace("_", "\n"))
        avgs.append(np.mean(vals))
        p95s.append(np.percentile(vals, 95))

    if not labels:
        return {}

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, avgs, width, label="Avg", color="#4C72B0")
    ax.bar(x + width / 2, p95s, width, label="P95", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Pipeline Step Latency (8 steps)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = figures_dir / "pipeline_steps_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"pipeline_steps_distribution": str(path)}


def _plot_judge_distribution(results: E7Results, figures_dir: Path) -> Dict[str, str]:
    if not results.judge_scores:
        return {}

    scores = list(results.judge_scores.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=min(15, len(scores)), color="#55A868", edgecolor="white")
    ax.set_xlabel("Judge score")
    ax.set_ylabel("Count")
    ax.set_title("LLM Judge Score Distribution")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    path = figures_dir / "judge_scores_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"judge_scores_distribution": str(path)}


def _plot_recall_precision_scatter(results: E7Results, figures_dir: Path) -> Dict[str, str]:
    recalls = []
    precisions = []
    for tc_acc in results.per_tc_accuracy.values():
        recalls.append(tc_acc.get("recall", 0.0))
        precisions.append(tc_acc.get("precision", 0.0))

    if not recalls:
        return {}

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(recalls, precisions, alpha=0.7, color="#8172B3")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Per-TC Precision vs Recall")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    path = figures_dir / "fallback_latency_impact.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"fallback_latency_impact": str(path)}


def _plot_latency_flame(results: E7Results, figures_dir: Path) -> Dict[str, str]:
    components = {
        "Parse (ASRAdapter)": "parse",
        "E2E Pipeline": "e2e",
        "Rule Evaluation": "RULE_EVALUATION",
    }
    labels = []
    values = []
    for label, key in components.items():
        stats = results.latency.get(key, {})
        avg = stats.get("avg_ms", 0.0)
        if avg > 0:
            labels.append(label)
            values.append(avg)

    if not values:
        return {}

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color="#C44E52")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Avg latency (ms)")
    ax.set_title("Pipeline Component Latency")
    ax.grid(axis="x", alpha=0.3)

    path = figures_dir / "latency_flame.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"latency_flame": str(path)}
