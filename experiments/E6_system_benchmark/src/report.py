import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from evaluator import E6Results


MACRO_COMPONENTS = {
    "BERT Parse": lambda b: b.bert_ms,
    "Native Rules": lambda b: sum(b.native_rules.values()) if b.native_rules else 0,
    "Compiled Rules": lambda b: sum(b.compiled_rules.values()) if b.compiled_rules else 0,
    "Generic LLM": lambda b: sum(b.generic_rules.values()) if b.generic_rules else 0,
    "Pipeline": lambda b: b.pipeline_e2e_ms,
}


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
        fig_paths.update(_plot_system_accuracy_boxplot(results, figures_dir))
        fig_paths.update(_plot_latency_stress_curves(results, figures_dir))
        fig_paths.update(_plot_pipeline_macro_components(results, figures_dir))
        fig_paths.update(_plot_fallback_impact(results, figures_dir))
        fig_paths.update(_plot_judge_distribution(results, figures_dir))

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


def _load_test_cases_meta(results) -> Dict[str, Any]:
    test_cases_path = Path(results.config.ground_truth_dir) / "test_cases.json"
    if not test_cases_path.exists():
        return {}
    with open(test_cases_path) as f:
        tcs = json.load(f)
    return {tc["id"]: tc for tc in tcs}


def _compute_per_tc_accuracy(results) -> Tuple[List[float], List[float], List[float]]:
    tc_meta = _load_test_cases_meta(results)
    precisions, recalls, f1s = [], [], []

    for bench in results.all_benchmarks:
        tc_info = tc_meta.get(bench.test_case_id)
        if not tc_info:
            continue

        expected = tc_info.get("expected_alerts", {})
        er = bench.eval_results

        tp = fp = tn = fn = 0
        for rule_id, exp in expected.items():
            actual = er.get(rule_id)
            if actual is None:
                continue
            if exp.get("satisfied", True):
                if actual.satisfied:
                    tn += 1
                else:
                    fp += 1
            else:
                if not actual.satisfied:
                    tp += 1
                else:
                    fn += 1

        total = tp + fp + tn + fn
        if total == 0:
            continue

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return precisions, recalls, f1s


def _plot_system_accuracy_boxplot(results, figures_dir):
    precisions, recalls, f1s = _compute_per_tc_accuracy(results)

    if not precisions:
        return {}

    data = [precisions, recalls, f1s]
    labels = ["Precision", "Recall", "F1"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    colors = ["#2ecc71", "#3498db", "#f39c12"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"E6: System-Level Detection Accuracy (n={len(precisions)} test cases)")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="Random baseline")
    ax.legend()

    means = [np.mean(d) for d in data]
    for i, m in enumerate(means):
        ax.text(i + 1, m + 0.03, f"\u03bc={m:.3f}", ha="center", fontsize=9, color="darkblue")

    fig.tight_layout()
    path = figures_dir / "system_overall_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"system_overall_accuracy": str(path)}


def _plot_latency_stress_curves(results, figures_dir):
    if not results.all_benchmarks:
        return {}

    test_cases_path = Path(results.config.ground_truth_dir) / "test_cases.json"
    if not test_cases_path.exists():
        return {}
    with open(test_cases_path) as f:
        tcs_raw = json.load(f)
    tc_meta = {tc["id"]: tc for tc in tcs_raw}

    eval_data = {"native+compiled": {"x": [], "y": []}, "generic": {"x": [], "y": []}}

    for bench in results.all_benchmarks:
        tc_info = tc_meta.get(bench.test_case_id, {})
        n_aircraft = len(tc_info.get("traffic_state", {}).get("aircrafts", []))
        n_rules = len(bench.compiled_rules) + len(bench.generic_rules)
        pipeline_ms = bench.pipeline_e2e_ms

        if pipeline_ms <= 0:
            continue

        eval_type = tc_info.get("eval_type", "native+compiled")
        eval_data[eval_type]["x"].append(n_rules)
        eval_data[eval_type]["y"].append(pipeline_ms)

    has_data = any(len(v["x"]) > 0 for v in eval_data.values())
    if not has_data:
        return {}

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"native+compiled": "#2ecc71", "generic": "#e74c3c"}

    for eval_type, data in eval_data.items():
        x = np.array(data["x"])
        y = np.array(data["y"])
        if len(x) < 2:
            if len(x) == 1:
                ax.scatter(x, y, alpha=0.5, s=30, color=colors.get(eval_type, "gray"),
                           label=f"{eval_type} (n=1)")
            continue

        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        ax.scatter(x, y, alpha=0.5, s=30, color=colors.get(eval_type, "gray"),
                   label=f"{eval_type} (n={len(x)})")

        if len(x) >= 3:
            z = np.polyfit(x_sorted, y_sorted, 2)
            p = np.poly1d(z)
            x_line = np.linspace(x_sorted.min(), x_sorted.max(), 100)
            ax.plot(x_line, p(x_line), color=colors.get(eval_type, "gray"),
                    linestyle="--", alpha=0.8, linewidth=2)

    ax.set_xlabel("# Active Rules Evaluated")
    ax.set_ylabel("Pipeline Latency (ms)")
    ax.set_title("E6: Latency vs Evaluation Complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = figures_dir / "latency_percentiles_stress.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"latency_percentiles_stress": str(path)}


def _plot_pipeline_macro_components(results, figures_dir):
    if not results.all_benchmarks:
        return {}

    macro_values = {name: [] for name in MACRO_COMPONENTS}

    for bench in results.all_benchmarks:
        for name, extractor in MACRO_COMPONENTS.items():
            val = extractor(bench)
            if val > 0:
                macro_values[name].append(val)

    macro_labels = []
    macro_avgs = []
    macro_p95s = []
    macro_counts = []
    for name in MACRO_COMPONENTS:
        vals = macro_values[name]
        if not vals:
            continue
        macro_labels.append(name)
        macro_avgs.append(np.mean(vals))
        s = sorted(vals)
        idx = int(len(s) * 0.95)
        macro_p95s.append(s[min(idx, len(s) - 1)])
        macro_counts.append(len(vals))

    if not macro_labels:
        return {}

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(macro_labels))
    h = 0.35
    ax.barh(y - h / 2, macro_avgs, h, label="Avg", color="#3498db", alpha=0.8)
    ax.barh(y + h / 2, macro_p95s, h, label="P95", color="#e74c3c", alpha=0.8)

    for i, (avg, p95, cnt) in enumerate(zip(macro_avgs, macro_p95s, macro_counts)):
        ax.text(avg + max(macro_p95s) * 0.02, i - h / 2, f"{avg:.2f}ms (n={cnt})", va="center", fontsize=9)
        ax.text(p95 + max(macro_p95s) * 0.02, i + h / 2, f"{p95:.2f}ms", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(macro_labels)
    ax.set_xlabel("Latency (ms)")
    ax.set_title("E6: Pipeline Macro-Component Breakdown")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    path = figures_dir / "pipeline_steps_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"pipeline_steps_distribution": str(path)}


def _plot_fallback_impact(results, figures_dir):
    if not results.all_benchmarks:
        return {}

    compiled_latencies = []
    generic_latencies = []

    for bench in results.all_benchmarks:
        tc = bench.compiled_rules
        if tc:
            compiled_latencies.append(sum(tc.values()))

        gr = bench.generic_rules
        if gr:
            generic_latencies.append(sum(gr.values()))

    if not compiled_latencies and not generic_latencies:
        return {}

    fig, ax = plt.subplots(figsize=(10, 5))

    if compiled_latencies:
        nc = np.array(compiled_latencies)
        nc = nc[nc > 0]
        if len(nc) > 1:
            try:
                kde_nc = gaussian_kde(nc)
                x_nc = np.linspace(nc.min(), nc.max(), 200)
                ax.fill_between(x_nc, kde_nc(x_nc), alpha=0.4, color="#2ecc71",
                                label=f"Native+Compiled (n={len(nc)}, \u03bc={np.mean(nc):.2f}ms)")
                ax.plot(x_nc, kde_nc(x_nc), color="#2ecc71", linewidth=2)
            except Exception:
                pass
        ax.hist(nc, bins=min(20, len(nc)), alpha=0.3, color="#2ecc71", density=True)

    if generic_latencies:
        gl = np.array(generic_latencies)
        gl = gl[gl > 0]
        if len(gl) > 1:
            try:
                kde_gl = gaussian_kde(gl)
                x_gl = np.linspace(gl.min(), gl.max(), 200)
                ax.fill_between(x_gl, kde_gl(x_gl), alpha=0.4, color="#e74c3c",
                                label=f"Generic LLM (n={len(gl)}, \u03bc={np.mean(gl):.2f}ms)")
                ax.plot(x_gl, kde_gl(x_gl), color="#e74c3c", linewidth=2)
            except Exception:
                pass
        ax.hist(gl, bins=min(20, len(gl)), alpha=0.3, color="#e74c3c", density=True)

    ax.set_xlabel("Rule Evaluation Latency (ms)")
    ax.set_ylabel("Density")
    ax.set_title("E6: Fallback Latency Impact (Deterministic vs LLM)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = figures_dir / "fallback_latency_impact.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"fallback_latency_impact": str(path)}


def _plot_judge_distribution(results, figures_dir):
    if not results.judge_scores:
        return {}

    scores = list(results.judge_scores.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    n, bins, patches = ax.hist(scores, bins=10, edgecolor="white", alpha=0.8, range=(0, 1))

    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge >= 0.7:
            patch.set_facecolor("#2ecc71")
        elif left_edge >= 0.5:
            patch.set_facecolor("#f39c12")
        else:
            patch.set_facecolor("#e74c3c")

    mean_score = np.mean(scores)
    median_score = np.median(scores)
    ax.axvline(mean_score, color="darkblue", linestyle="--", linewidth=2, label=f"Mean={mean_score:.3f}")
    ax.axvline(median_score, color="orange", linestyle="--", linewidth=2, label=f"Median={median_score:.3f}")

    ax.set_xlabel("Judge Score")
    ax.set_ylabel("Frequency")
    ax.set_title(f"E6: Semantic Judge Score Distribution (n={len(scores)})")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = figures_dir / "judge_scores_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"judge_scores_distribution": str(path)}
