import json
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluator import E5Results


def generate_report(
    results: E5Results,
    cfg,
    save_figures: bool = True,
) -> Dict[str, Any]:
    results_dir = cfg.results_dir
    figures_dir = cfg.figures_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    strategies = list(results.overall_scores.keys())
    n = len(strategies)

    fig_paths = {}

    if save_figures and n > 0:
        fig_paths.update(
            _plot_overall_scores(results, figures_dir, strategies)
        )
        fig_paths.update(
            _plot_metric_breakdown(results, figures_dir, strategies)
        )
        fig_paths.update(
            _plot_precision_recall_f1(results, figures_dir, strategies)
        )
        fig_paths.update(
            _plot_severity_accuracy(results, figures_dir, strategies)
        )
        fig_paths.update(
            _plot_latency(results, figures_dir, strategies)
        )
        fig_paths.update(
            _plot_judge_scores(results, figures_dir, strategies)
        )

    ranking = results.ranking
    summary = {
        "ranking": ranking,
        "overall_scores": results.overall_scores,
        "details": {},
    }

    for entry in ranking:
        strat = entry["strategy"]
        sm = results.strategies.get(strat)
        js = results.judge_scores.get(strat, {})
        sem_avg = sum(js.values()) / len(js) if js else 0.0
        per_rule = {}
        if sm:
            for rid, rm in sm.per_rule.items():
                per_rule[rid] = {
                    "precision": round(rm.precision, 4),
                    "recall": round(rm.recall, 4),
                    "f1": round(rm.f1, 4),
                    "accuracy": round(rm.accuracy, 4),
                    "severity_accuracy": round(rm.severity_accuracy, 4),
                    "tp": rm.true_positives,
                    "fp": rm.false_positives,
                    "tn": rm.true_negatives,
                    "fn": rm.false_negatives,
                }
        summary["details"][strat] = {
            "per_rule": per_rule,
            "avg_latency_ms": round(sm.avg_latency_ms, 2) if sm else 0.0,
            "semantic_scores": {k: round(v, 4) for k, v in js.items()},
        }

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    result_path = results_dir / "results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ranking": ranking,
                "overall_scores": results.overall_scores,
            },
            f,
            indent=2,
        )

    return {
        "results": str(result_path),
        "summary": str(summary_path),
        "figures": fig_paths,
        "ranking": ranking,
    }


def _plot_overall_scores(results, figures_dir, strategies):
    scores = [results.overall_scores[s] for s in strategies]
    colors = ["#4ECDC4", "#FF6B6B"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(strategies, scores, color=colors[: len(strategies)], width=0.5)
    for bar, v in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Overall Score")
    ax.set_title("E5: Overall Score Comparison")
    ax.set_ylim(0, max(scores) * 1.15 if scores else 1.0)
    fig.tight_layout()
    path = figures_dir / "overall_score_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"overall_score_comparison": str(path)}


def _plot_metric_breakdown(results, figures_dir, strategies):
    metrics = ["Precision", "Recall", "F1", "Sev. Acc."]
    x = np.arange(len(metrics))
    width = 0.3
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, strat in enumerate(strategies):
        sm = results.strategies[strat]
        vals = [
            sm.overall_precision,
            sm.overall_recall,
            sm.overall_f1,
            sm.overall_severity_accuracy,
        ]
        offset = (i - (len(strategies) - 1) / 2) * width
        bars = ax.bar(
            x + offset, vals, width, label=strat,
            color=["#4ECDC4", "#FF6B6B"][i],
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("E5: Metric Breakdown by Strategy")
    ax.legend()
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    path = figures_dir / "metric_breakdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"metric_breakdown": str(path)}


def _plot_precision_recall_f1(results, figures_dir, strategies):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Precision", "Recall", "F1 Score"]

    for idx, (ax, title) in enumerate(zip(axes, titles)):
        for i, strat in enumerate(strategies):
            sm = results.strategies[strat]
            vals = [getattr(rm, ["precision", "recall", "f1"][idx]) for rm in sm.per_rule.values()]
            rule_ids = list(sm.per_rule.keys())
            x = np.arange(len(vals))
            ax.bar(
                x + i * 0.25,
                vals,
                0.25,
                label=strat if idx == 0 else "",
                color=["#4ECDC4", "#FF6B6B"][i],
            )
            for xi, v in zip(x, vals):
                ax.text(
                    xi + i * 0.25, v + 0.01, f"{v:.2f}",
                    ha="center", fontsize=7,
                )
        ax.set_xticks(x + 0.125)
        ax.set_xticklabels(rule_ids)
        ax.set_title(title)
        ax.set_ylim(0, 1.15)
        if idx == 0:
            ax.legend()
    fig.tight_layout()
    path = figures_dir / "precision_recall_f1.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"precision_recall_f1": str(path)}


def _plot_severity_accuracy(results, figures_dir, strategies):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(strategies))
    width = 0.4
    for i, strat in enumerate(strategies):
        sm = results.strategies[strat]
        overall_sa = sm.overall_severity_accuracy
        ax.bar(i, overall_sa, width, color=["#4ECDC4", "#FF6B6B"][i], label=strat)
        ax.text(i, overall_sa + 0.01, f"{overall_sa:.2%}", ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylabel("Severity Accuracy")
    ax.set_title("E5: Severity Classification Accuracy")
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    path = figures_dir / "severity_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"severity_accuracy": str(path)}


def _plot_latency(results, figures_dir, strategies):
    fig, ax = plt.subplots(figsize=(6, 4))
    latencies = [results.strategies[s].avg_latency_ms for s in strategies]
    colors = ["#4ECDC4", "#FF6B6B"]
    bars = ax.bar(strategies, latencies, color=colors[: len(strategies)], width=0.5)
    for bar, v in zip(bars, latencies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(latencies) * 0.02,
            f"{v:.0f}ms",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Avg Latency (ms)")
    ax.set_title("E5: Evaluation Latency Comparison")
    fig.tight_layout()
    path = figures_dir / "latency_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"latency_comparison": str(path)}


def _plot_judge_scores(results, figures_dir, strategies):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(strategies))
    for i, strat in enumerate(strategies):
        js = results.judge_scores.get(strat, {})
        scores = list(js.values())
        if scores:
            mean_val = np.mean(scores)
            std_val = np.std(scores)
            ax.bar(
                i,
                mean_val,
                0.4,
                yerr=std_val,
                capsize=5,
                color=["#4ECDC4", "#FF6B6B"][i],
                label=strat,
            )
            ax.text(i, mean_val + 0.02, f"{mean_val:.3f}", ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylabel("Mean Judge Score")
    ax.set_title("E5: Semantic Judge Quality Scores")
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    path = figures_dir / "judge_scores.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"judge_scores": str(path)}
