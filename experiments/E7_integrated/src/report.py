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
        fig_paths.update(_plot_tp_fp_fn_stacked_horizontal(results, figures_dir))
        fig_paths.update(_plot_tp_fp_fn_heatmap(results, figures_dir))
        fig_paths.update(_plot_tp_fp_fn_zoom_distribution(results, figures_dir))

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


def _plot_tp_fp_fn_stacked_horizontal(results: E7Results, figures_dir: Path) -> Dict[str, str]:
    per_tc = results.per_tc_accuracy
    if not per_tc:
        return {}

    items = [(tc_id, m.get("tp", 0), m.get("fp", 0), m.get("fn", 0))
             for tc_id, m in per_tc.items()]
    items.sort(key=lambda x: x[1] + x[2] + x[3], reverse=True)

    tc_ids = [x[0] for x in items]
    tps = [x[1] for x in items]
    fps = [x[2] for x in items]
    fns = [x[3] for x in items]

    y_pos = np.arange(len(tc_ids))
    bar_h = 0.5

    fig, ax = plt.subplots(figsize=(12, 11))
    ax.barh(y_pos, tps, bar_h, label="TP", color="#2ca02c")
    ax.barh(y_pos, fns, bar_h, left=tps, label="FN", color="#ff7f0e")
    ax.barh(y_pos, fps, bar_h, left=[t + fn for t, fn in zip(tps, fns)],
            label="FP", color="#d62728")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tc_ids, fontsize=6)
    ax.set_xlabel("Alert count")
    ax.set_title("TP / FN / FP per Test Case (sorted by total errors)")
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    path = figures_dir / "tp_fp_fn_stacked_per_tc.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"tp_fp_fn_stacked_per_tc": str(path)}


def _plot_tp_fp_fn_heatmap(results: E7Results, figures_dir: Path) -> Dict[str, str]:
    per_tc = results.per_tc_accuracy
    if not per_tc:
        return {}

    tc_ids = sorted(per_tc.keys())
    n = len(tc_ids)
    data = np.zeros((n, 3), dtype=int)
    for i, tc_id in enumerate(tc_ids):
        m = per_tc[tc_id]
        data[i, 0] = m.get("tp", 0)
        data[i, 1] = m.get("fp", 0)
        data[i, 2] = m.get("fn", 0)

    vmax = data.max()
    fig, ax = plt.subplots(figsize=(6, n * 0.28 + 1))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(vmax, 1))

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Count")

    ax.set_xticks(range(3))
    ax.set_xticklabels(["TP", "FP", "FN"], fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tc_ids, fontsize=5)

    for i in range(n):
        for j in range(3):
            val = data[i, j]
            color = "white" if val > vmax * 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=4.5, color=color)

    ax.set_title("TP / FP / FN Heatmap (50 TCs)", fontsize=10)
    fig.tight_layout()
    path = figures_dir / "tp_fp_fn_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"tp_fp_fn_heatmap": str(path)}


def _plot_tp_fp_fn_zoom_distribution(results: E7Results, figures_dir: Path) -> Dict[str, str]:
    per_tc = results.per_tc_accuracy
    if not per_tc:
        return {}

    items = [(tc_id, m.get("tp", 0), m.get("fp", 0), m.get("fn", 0))
             for tc_id, m in per_tc.items()]

    problematic = [(tc_id, tp, fp, fn) for tc_id, tp, fp, fn in items if fp > 5 or fn > 0]
    problematic.sort(key=lambda x: x[1] + x[2] + x[3], reverse=True)

    all_fps = [fp for _, _, fp, _ in items]
    all_fns = [fn for _, _, _, fn in items]
    max_fn = max(all_fns) if all_fns else 0
    max_fp_bin = max(all_fps) if all_fps else 10

    bin_edges = list(range(0, max_fp_bin + 2, max(1, (max_fp_bin + 1) // 15)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7),
                                    gridspec_kw={"width_ratios": [1, 1]})

    if problematic:
        tc_ids_p = [x[0] for x in problematic]
        tps_p = [x[1] for x in problematic]
        fps_p = [x[2] for x in problematic]
        fns_p = [x[3] for x in problematic]
        y_pos_p = np.arange(len(tc_ids_p))
        bar_h = 0.5

        ax1.barh(y_pos_p, tps_p, bar_h, label="TP", color="#2ca02c")
        ax1.barh(y_pos_p, fns_p, bar_h, left=tps_p, label="FN", color="#ff7f0e")
        ax1.barh(y_pos_p, fps_p, bar_h,
                 left=[t + fn for t, fn in zip(tps_p, fns_p)],
                 label="FP", color="#d62728")
        ax1.set_yticks(y_pos_p)
        ax1.set_yticklabels(tc_ids_p, fontsize=7)
        ax1.invert_yaxis()
        ax1.set_xlabel("Alert count")
        ax1.set_title(f"TCs with FP>5 or FN>0 ({len(problematic)} TCs)")
        ax1.legend(loc="lower right")
        ax1.grid(axis="x", alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No problematic TCs", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Zoom: problematic TCs")

    ax2.hist(all_fps, bins=bin_edges, alpha=0.7, label="FP", color="#d62728", edgecolor="white")
    ax2.hist(all_fns, bins=bin_edges, alpha=0.7, label="FN", color="#ff7f0e", edgecolor="white")
    ax2.set_xlabel("Count per TC")
    ax2.set_ylabel("Number of TCs")
    ax2.set_title("Distribution of FP and FN across TCs")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = figures_dir / "tp_fp_fn_zoom_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"tp_fp_fn_zoom_distribution": str(path)}
