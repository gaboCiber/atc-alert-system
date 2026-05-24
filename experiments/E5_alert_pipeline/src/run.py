#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from experiments.E5_alert_pipeline.src.config import (
    E5Config,
    JudgeConfig,
    GenericConfig,
    MetricConfig,
)
from experiments.E5_alert_pipeline.src.loader import ExperimentData
from experiments.E5_alert_pipeline.src.semantic_judge import SemanticJudge
from experiments.E5_alert_pipeline.src.evaluator import run_evaluation
from experiments.E5_alert_pipeline.src.report import generate_report


def main():
    parser = argparse.ArgumentParser(
        description="E5: End-to-End Alert Pipeline Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                                                # default paths
  python run.py --no-judge                                     # skip semantic judge
  python run.py --judge-model gemma4:31b-cloud
  python run.py --generic-model gemma4:31b-cloud
        """,
    )

    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--gt-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--compiled-rules-dir", type=str, default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--judge-model", type=str, default="gemma4:31b-cloud")
    parser.add_argument(
        "--judge-provider",
        type=str,
        default="openai",
        choices=["openai", "ollama", "gemini", "anthropic"],
    )
    parser.add_argument(
        "--judge-base-url", type=str, default="http://localhost:11434/v1"
    )
    parser.add_argument("--judge-api-key", type=str, default="ollama")
    parser.add_argument("--generic-model", type=str, default="gemma4:31b-cloud")
    parser.add_argument(
        "--generic-provider",
        type=str,
        default="openai",
        choices=["openai", "ollama", "gemini", "anthropic"],
    )
    parser.add_argument(
        "--generic-base-url", type=str, default="http://localhost:11434/v1"
    )

    args = parser.parse_args()

    judge_provider = args.judge_provider
    if judge_provider == "ollama":
        judge_provider = "openai"
    generic_provider = args.generic_provider
    if generic_provider == "ollama":
        generic_provider = "openai"

    cfg = E5Config.from_dirs(
        base_dir=args.base_dir,
        ground_truth_dir=args.gt_dir,
        output_dir=args.output,
        compiled_rules_dir=args.compiled_rules_dir,
    )

    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    judge_cfg = JudgeConfig(
        model_name=args.judge_model,
        provider=judge_provider,
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
        enabled=not args.no_judge,
    )

    generic_cfg = GenericConfig(
        model_name=args.generic_model,
        provider=generic_provider,
        base_url=args.generic_base_url,
        api_key="ollama",
    )

    metric_cfg = MetricConfig()

    print("=" * 60)
    print("E5: End-to-End Alert Pipeline Comparison")
    print("=" * 60)
    print(f"  Ground truth:      {cfg.ground_truth_dir}")
    print(f"  Compiled rules:    {cfg.compiled_rules_dir}")
    print(f"  Output:            {cfg.results_dir}")
    print(f"  Semantic Judge:    {'enabled' if judge_cfg.enabled else 'disabled'}")
    if judge_cfg.enabled:
        print(f"    Judge Model:     {judge_cfg.model_name}")
        print(f"    Generic Model:   {generic_cfg.model_name}")
    print()

    print("Loading experiment data...")
    try:
        data = ExperimentData.from_config(cfg)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"  Strategies:  {', '.join(data.strategy_names)}")
    print(f"  Test cases:  {len(data.test_cases)}")
    print(f"  Rules:       {', '.join(data.relevant_rule_ids)}")
    print(f"  Compiled:    {len(data.compiled_rules)} rules loaded")

    if judge_cfg.enabled:
        print(f"  Descriptions: {len(data.rule_descriptions)} rules")
    print()

    judge = SemanticJudge(judge_cfg)

    print("Running evaluation...")
    results = run_evaluation(data, judge, generic_cfg, metric_cfg)
    print()

    print("Generating report...")
    report = generate_report(results, cfg, save_figures=not args.no_figures)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Results: {report['results']}")
    print(f"  Summary: {report['summary']}")
    if not args.no_figures:
        print("  Figures:")
        for name in report["figures"]:
            print(f"    - {name}.png")

    print()
    print("STRATEGY RANKING (by overall score):")
    for entry in report.get("ranking", []):
        print(f"  #{entry['rank']}: {entry['strategy']}")
        print(
            f"       Score={entry['overall_score']:.3f} | "
            f"P={entry['alert_precision']:.2%} | "
            f"R={entry['alert_recall']:.2%} | "
            f"F1={entry['alert_f1']:.2%} | "
            f"Sev={entry['severity_accuracy']:.2%} | "
            f"Sem={entry['semantic_score']:.3f} | "
            f"Lat={entry['avg_latency_ms']:.0f}ms"
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
