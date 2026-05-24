import json
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluator import E6Results


def generate_report(
    results: E6Results,
    cfg,
    save_figures: bool = True,
) -> Dict[str, Any]:
    results_dir = cfg.results_dir
    figures_dir = cfg.figures_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig_paths = {}

    if save_figures:
        fig_paths.update(_plot_latency_flame(results, figures_dir))
        fig_paths.update(_plot_latency_breakdown(results, figures_dir))
        fig_paths.update(_plot_latency_per_tc(results, figures_dir))
        fig_paths.update(_plot_accuracy(results, figures_dir))
        fig_paths.update(_plot_pipeline_steps(results, figures_dir))
        fig_paths.update(_plot_judge_scores(results, figures_dir))

    summary = {
        "latency": {k: v.to_dict() for k, v in results.latency.items()},
        "accuracy": {
            rid: {
                "precision": round(rm.precision, 4),
                "recall": round(rm.recall, 4),
                "f1": round(rm.f1, 4),
                "severity_accuracy": round(rm.severity_accuracy, 4),
                "tp": rm.true_positives,
                "fp": rm.false_positives,
                "tn": rm.true_negatives,
                "fn": rm.false_negatives,
            }
            for rid, rm in results.accuracy.items()
        },
        "judge_scores": results.judge_scores,
        "ranking": results.ranking,
    }

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "results": str(results_dir),
        "summary": str(summary_path),
        "figures": fig_paths,
    }


def _plot_latency_flame(results, figures_dir):
    components = ["bert", "native", "compiled", "generic", "pipeline"]
    labels = []
    values = []
    for c in components:
        ls = results.latency.get(c)
        if ls and ls.avg_ms > 0:
            labels.append(c.replace("_", " ").title())
            values.append(ls.avg_ms)

    if not values:
        return {}

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]
    bars = ax.barh(range(len(values)), values, color=colors[:len(values)])
    for i, (bar, v) in enumerate(zip(bars, values)):
        ax.text(bar.get_width() + max(values) * 0.02, i, f"{v:.0f}ms", va="center", fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Avg Latency (ms)")
    ax.set_title("E6: Latency Flame Graph (avg by component)")
    ax.invert_yaxis()

    # Add latency thresholds
    for threshold, color, label in [(500, "orange", "Warning"), (2000, "red", "Critical")]:
        if max(values) > threshold:
            ax.axvline(threshold, color=color, linestyle="--", alpha=0.5, label=f"{label} {threshold}ms")
    if max(values) > 500:
        ax.legend()

    fig.tight_layout()
    path = figures_dir / "latency_flame.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"latency_flame": str(path)}


def _plot_latency_breakdown(results, figures_dir):
    step_names = []
    avg_vals = []
    p95_vals = []
    for k, ls in sorted(results.latency.items()):
        if ls.avg_ms > 0 and k not in ("bert", "native", "compiled", "generic", "pipeline"):
            step_names.append(k)
            avg_vals.append(ls.avg_ms)
            p95_vals.append(ls.p95_ms)

    if not step_names:
        return {}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(step_names))
    w = 0.35
    ax.bar(x - w / 2, avg_vals, w, label="Avg", color="#3498db")
    ax.bar(x + w / 2, p95_vals, w, label="P95", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(step_names, rotation=45, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("E6: Latency Breakdown by Rule (Avg vs P95)")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "latency_breakdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"latency_breakdown": str(path)}


def _plot_latency_per_tc(results, figures_dir):
    if not results.all_benchmarks:
        return {}

    tc_ids = [b.test_case_id for b in results.all_benchmarks]
    bert = [b.bert_ms for b in results.all_benchmarks]
    native = [sum(b.native_rules.values()) for b in results.all_benchmarks]
    compiled = [sum(b.compiled_rules.values()) for b in results.all_benchmarks]
    generic = [sum(b.generic_rules.values()) for b in results.all_benchmarks]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(tc_ids))
    w = 0.2
    b1 = ax.bar(x - 1.5 * w, bert, w, label="BERT", color="#3498db")
    b2 = ax.bar(x - 0.5 * w, native, w, label="Native", color="#2ecc71")
    b3 = ax.bar(x + 0.5 * w, compiled, w, label="Compiled", color="#f39c12")
    b4 = ax.bar(x + 1.5 * w, generic, w, label="Generic LLM", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(tc_ids, rotation=45, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("E6: Latency per Test Case by Component")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "latency_per_tc.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"latency_per_tc": str(path)}


def _plot_accuracy(results, figures_dir):
    if not results.accuracy:
        return {}

    rule_ids = sorted(results.accuracy.keys())
    prec = [results.accuracy[r].precision for r in rule_ids]
    rec = [results.accuracy[r].recall for r in rule_ids]
    f1 = [results.accuracy[r].f1 for r in rule_ids]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(rule_ids))
    w = 0.25
    ax.bar(x - w, prec, w, label="Precision", color="#2ecc71")
    ax.bar(x, rec, w, label="Recall", color="#3498db")
    ax.bar(x + w, f1, w, label="F1", color="#f39c12")
    for i, (p, r, f) in enumerate(zip(prec, rec, f1)):
        ax.text(i - w, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
        ax.text(i + w, f + 0.02, f"{f:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(rule_ids)
    ax.set_ylabel("Score")
    ax.set_title("E6: Alert Accuracy by Rule")
    ax.set_ylim(0, 1.15)
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"accuracy": str(path)}


def _plot_pipeline_steps(results, figures_dir):
    all_step_names = sorted(
        {s["name"] for steps in results.pipeline_steps.values() if steps
         for s in steps.values()}
    ) if results.pipeline_steps else []

    if not all_step_names:
        return {}

    fig, ax = plt.subplots(figsize=(10, 5))
    tc_ids = list(results.pipeline_steps.keys())
    x = np.arange(len(tc_ids))
    w = 0.8 / max(len(all_step_names), 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_step_names)))

    bottom = np.zeros(len(tc_ids))
    for i, sname in enumerate(all_step_names):
        vals = []
        for tc_id in tc_ids:
            steps = results.pipeline_steps.get(tc_id, {})
            val = sum(
                s["execution_time_ms"]
                for s in steps.values()
                if s["name"] == sname
            )
            vals.append(val)
        ax.bar(x, vals, w, bottom=bottom, label=sname, color=colors[i])
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(tc_ids, rotation=45, ha="right")
    ax.set_ylabel("Execution Time (ms)")
    ax.set_title("E6: Pipeline Step Breakdown (stacked)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = figures_dir / "pipeline_steps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"pipeline_steps": str(path)}


def _plot_judge_scores(results, figures_dir):
    if not results.judge_scores:
        return {}

    fig, ax = plt.subplots(figsize=(8, 4))
    tc_ids = list(results.judge_scores.keys())
    scores = list(results.judge_scores.values())
    colors = ["#2ecc71" if s >= 0.7 else "#f39c12" if s >= 0.5 else "#e74c3c" for s in scores]
    bars = ax.bar(tc_ids, scores, color=colors)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(tc_ids)))
    ax.set_xticklabels(tc_ids, rotation=45, ha="right")
    ax.set_ylabel("Judge Score")
    ax.set_title("E6: Semantic Judge Scores per Test Case")
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    path = figures_dir / "judge_scores.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"judge_scores": str(path)}
