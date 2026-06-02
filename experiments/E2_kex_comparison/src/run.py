#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import E2Config, JudgeConfig, MetricConfig, DedupConfig
from loader import ExperimentData
from llm_judge import LLMJudge
from evaluator import run_evaluation
from report import generate_report
from dedup import format_card


def main():
    parser = argparse.ArgumentParser(
        description="E2: KEX comparison between models and Ground Truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                                                  # default paths
  python run.py --no-judge                                       # skip LLM judge
  python run.py --judge-model gpt-4o --judge-provider openai     # custom judge
  python run.py --gt-dir /path/to/gt --models-dir /path/to/models
        """,
    )

    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--gt-dir", type=str, default=None)
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-a-judge evaluation")
    parser.add_argument("--judge-model", type=str, default="llama3.2", help="LLM model for judging")
    parser.add_argument("--judge-provider", type=str, default="openai", choices=["openai", "ollama", "gemini", "anthropic"])
    parser.add_argument("--judge-base-url", type=str, default="http://localhost:11434/v1")
    parser.add_argument("--judge-api-key", type=str, default="ollama")
    parser.add_argument("--judge-max-retries", type=int, default=3)
    parser.add_argument("--fuzzy-threshold", type=float, default=70.0)
    parser.add_argument("--semantic-weight", type=float, default=0.60)
    parser.add_argument("--no-dedup", action="store_true", help="Skip dedup analysis")
    parser.add_argument("--dedup-batch-size", type=int, default=10, help="Batch size for dedup LLM calls")
    parser.add_argument("--dedup-threshold", type=float, default=0.80, help="Similarity threshold for dedup")

    args = parser.parse_args()

    cfg = E2Config.from_dirs(
        base_dir=args.base_dir,
        ground_truth_dir=args.gt_dir,
        models_dir=args.models_dir,
        output_dir=args.output,
    )

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

    metric_cfg = MetricConfig(
        semantic_weight=args.semantic_weight,
        fuzzy_threshold=args.fuzzy_threshold,
    )

    dedup_cfg = DedupConfig(
        enabled=not args.no_dedup and not args.no_judge,
        batch_size=args.dedup_batch_size,
        threshold=args.dedup_threshold,
    )

    print("=" * 60)
    print("E2: KEX Comparison Experiment")
    print("=" * 60)
    print(f"  Ground truth: {cfg.ground_truth_dir}")
    print(f"  Models:      {cfg.models_dir}")
    print(f"  Output:      {cfg.results_dir}")
    print(f"  LLM Judge:   {'enabled' if judge_cfg.enabled else 'disabled'}")
    if judge_cfg.enabled:
        print(f"    Model:     {judge_cfg.model_name}")
        print(f"    Provider:  {judge_cfg.provider}")
    print(f"  Weights:     Structural={metric_cfg.structural_weight}, Content={metric_cfg.content_weight}, CrossRef={metric_cfg.cross_ref_weight}, Semantic={metric_cfg.semantic_weight}")
    print(f"  Dedup:       {'enabled' if dedup_cfg.enabled else 'disabled'}")
    if dedup_cfg.enabled:
        print(f"    Batch:    {dedup_cfg.batch_size}")
        print(f"    Thresh:   {dedup_cfg.threshold}")
    print()

    print("Loading experiment data...")
    try:
        data = ExperimentData.from_config(cfg)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"  Found {len(data.model_names)} model(s):")
    for mn in data.model_names:
        print(f"    - {mn}")
    print(f"  Pages: {data.pages}")
    print()

    judge = LLMJudge(judge_cfg)

    print("Running evaluation...")
    results = run_evaluation(data, judge, metric_cfg, dedup_cfg)
    print(f"  Evaluated {len(data.pages)} pages across {len(data.model_names)} models")
    print()

    print("Generating report...")
    report = generate_report(
        results,
        cfg,
        save_figures=not args.no_figures,
    )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Results: {report['page_metrics']}")
    print(f"  Summary: {report['summary']}")
    if not args.no_figures:
        print("  Figures:")
        for name, path in report["figures"].items():
            print(f"    - {name}.png")

    print()
    print("MODEL RANKING (by overall score):")
    for entry in report.get("ranking", []):
        print(f"  #{entry['rank']}: {entry['model']}")
        print(f"       Score={entry['overall_score']:.3f} | F1={entry['structural_f1']:.3f} | Content={entry['content_avg']:.3f} | CrossRef={entry['cross_ref_avg']:.3f} | Semantic={entry['semantic_avg']:.3f}")

    print()
    print("=" * 60)
    print("DEDUP REPORT")
    print("=" * 60)
    if results.dedup_reports:
        for model_name, report in results.dedup_reports.items():
            card = format_card(report)
            dedup_card_path = cfg.results_dir / f"dedup_{model_name}_card.txt"
            dedup_card_path.write_text(card, encoding="utf-8")
            print(f"\n{model_name}:")
            for kex_type, m in report.by_type.items():
                print(f"  {kex_type:15s} {m.duplication_rate:>7.2%}  ({m.duplicate_count}/{m.total_items})  clusters: {m.cluster_count}")
            print(f"  {'OVERALL':15s} {report.overall_duplication_rate:>7.2%}")
            print(f"  (detailed card saved to {dedup_card_path})")
    else:
        print("  (disabled or no data)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
