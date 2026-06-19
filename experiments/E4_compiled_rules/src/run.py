#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import E4Config, JudgeConfig, MetricConfig
from loader import ExperimentData
from semantic_judge import SemanticJudge
from evaluator import run_evaluation
from report import generate_report


def main():
    parser = argparse.ArgumentParser(
        description="E4: Compiled Rules Generation Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                                              # default paths
  python run.py --no-judge                                  # skip semantic judge
  python run.py --judge-model gpt-4o --judge-provider openai
        """,
    )

    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--gt-dir", type=str, default=None)
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--judge-model", type=str, default="llama3.2")
    parser.add_argument("--judge-provider", type=str, default="openai", choices=["openai", "ollama", "gemini", "anthropic"])
    parser.add_argument("--judge-base-url", type=str, default="http://localhost:11434/v1")
    parser.add_argument("--judge-api-key", type=str, default="ollama")
    parser.add_argument("--judge-max-retries", type=int, default=3)
    parser.add_argument("--no-judge-cache", action="store_true", help="Skip judge cache, re-evaluate all rules")
    parser.add_argument("--judge-cache-dir", type=str, default=None, help="Directory for judge cache files (default: results/)")

    args = parser.parse_args()

    cfg = E4Config.from_dirs(
        base_dir=args.base_dir,
        ground_truth_dir=args.gt_dir,
        models_dir=args.models_dir,
        output_dir=args.output,
        judge_cache_dir=args.judge_cache_dir,
    )

    if args.no_judge_cache:
        cfg.judge_cache_dir = None

    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    judge_provider = args.judge_provider
    if judge_provider == "ollama":
        judge_provider = "openai"

    judge_cfg = JudgeConfig(
        model_name=args.judge_model,
        provider=judge_provider,
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
        max_retries=args.judge_max_retries,
        enabled=not args.no_judge,
    )

    metric_cfg = MetricConfig()

    print("=" * 60)
    print("E4: Compiled Rules Generation Comparison")
    print("=" * 60)
    print(f"  Ground truth:    {cfg.ground_truth_dir}")
    print(f"  Models:          {cfg.models_dir}")
    print(f"  Output:          {cfg.results_dir}")
    print(f"  Semantic Judge:  {'enabled' if judge_cfg.enabled else 'disabled'}")
    if judge_cfg.enabled:
        print(f"    Model:         {judge_cfg.model_name}")
        print(f"    Provider:     {judge_cfg.provider}")
        print(f"    Cache:         {'disabled' if cfg.judge_cache_dir is None else cfg.judge_cache_dir}")
    print()

    print("Loading experiment data...")
    try:
        data = ExperimentData.from_config(cfg)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"  Found {len(data.model_names)} model(s):")
    for mn in data.model_names:
        mr = data.model_results[mn]
        print(f"    - {mn}: {len(mr.compiled_rules)} compiled, {len(mr.failed_rules)} failed")
    print(f"  Ground truth rules: {len(data.ground_truth_classification)}")
    print(f"  Reference code:    {len(data.reference_code)} rules")
    print(f"  Test states:       {len(data.test_traffic_states)} states")
    print()

    judge = SemanticJudge(judge_cfg, cache_dir=cfg.judge_cache_dir)

    print("Running evaluation...")
    results = run_evaluation(data, judge, metric_cfg)
    print(f"  Evaluated {len(data.model_names)} models")
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
        for name, path in report["figures"].items():
            print(f"    - {name}.png")

    print()
    print("MODEL RANKING (by overall score):")
    for entry in report.get("ranking", []):
        print(f"  #{entry['rank']}: {entry['model']}")
        print(f"       Score={entry['overall_score']:.3f} | Cls={entry['classification_accuracy']:.2%} | Val={entry['validation_pass_rate']:.2%} | Exec={entry['execution_match_rate']:.2%} | Sem={entry['semantic_score']:.3f}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()