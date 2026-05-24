import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config import E6Config, LLMConfig
from evaluator import run_benchmark
from report import generate_report
from semantic_judge import SemanticJudge

RULE_DESCRIPTIONS: dict = {
    "RULE001": (
        "Altitude Rule: All aircraft must maintain at least 1000ft separation from MSA. "
        "An aircraft violates this rule if its altitude is less than MSA - 1000ft."
    ),
    "RULE002": (
        "Separation Rule: All aircraft pairs must maintain at least 3 NM horizontal "
        "and 1000ft vertical separation. A violation occurs if any pair is closer."
    ),
    "RULE004": (
        "Runway Rule: No aircraft may be on a runway that is occupied by another aircraft. "
        "A violation occurs if an aircraft is within 3 NM of an occupied runway."
    ),
}


def main():
    parser = argparse.ArgumentParser(description="E6: System-level Latency & Accuracy Benchmark")
    parser.add_argument("--base-dir", type=str, default=None, help="Override base dir")
    parser.add_argument("--ground-truth-dir", type=str, default=None, help="Override ground truth dir")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--compiled-rules-dir", type=str, default=None, help="Override compiled rules dir")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge evaluation")
    parser.add_argument("--model", type=str, default="gemma4:31b-cloud", help="LLM model for generic + judge")
    parser.add_argument("--provider", type=str, default="openai", help="Provider (openai, gemini, anthropic)")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434/v1", help="LLM API base URL")
    parser.add_argument("--api-key", type=str, default="ollama", help="LLM API key")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--measure", type=int, default=15, help="Measurement iterations")
    args = parser.parse_args()

    cfg = E6Config.from_dirs(
        base_dir=args.base_dir,
        ground_truth_dir=args.ground_truth_dir,
        output_dir=args.output_dir,
        compiled_rules_dir=args.compiled_rules_dir,
    )
    cfg.warmup_iterations = args.warmup
    cfg.measure_iterations = args.measure

    print(f"E6 System Benchmark")
    print(f"  Ground truth: {cfg.ground_truth_dir}")
    print(f"  Compiled rules: {cfg.compiled_rules_dir}")
    print(f"  LLM: {args.model} ({args.provider})")
    print(f"  Iterations: {cfg.warmup_iterations} warmup + {cfg.measure_iterations} measure")
    if args.no_judge:
        print("  Judge: disabled")
    print()

    llm_cfg = LLMConfig(
        model_name=args.model,
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    judge = None
    if not args.no_judge:
        judge = SemanticJudge(llm_cfg)

    results = run_benchmark(cfg, llm_cfg, judge, RULE_DESCRIPTIONS)

    report_info = generate_report(results, cfg, save_figures=True)

    print(f"Report: {report_info['results']}")
    print(f"Summary: {report_info['summary']}")
    print(f"Figures: {len(report_info['figures'])} generated")

    if results.judge_scores:
        avg_judge = sum(results.judge_scores.values()) / len(results.judge_scores)
        print(f"Avg judge score: {avg_judge:.3f}")

    # Accuracy summary
    for rule_id, rm in results.accuracy.items():
        print(f"  {rule_id}: P={rm.precision:.2f} R={rm.recall:.2f} F1={rm.f1:.2f}")

    # Ranking
    if results.ranking:
        for entry in results.ranking:
            print(f"  {entry['component']}: {entry['avg_latency_ms']}ms")

    return results


if __name__ == "__main__":
    main()
