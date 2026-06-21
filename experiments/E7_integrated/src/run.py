#!/usr/bin/env python3
"""CLI entry point para E7_integrated — Pipeline Benchmark Integrado."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import E7Config, LLMConfig, FilterConfig, BERTConfig
from evaluator import run_benchmark
from report import generate_report
from judge import PipelineJudge


def main():
    parser = argparse.ArgumentParser(
        description="E7: Integrated Alert_System Pipeline Benchmark"
    )
    parser.add_argument("--ground-truth-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--compiled-rules-dir", type=str, default=None)
    parser.add_argument("--rules-json", type=str, default=None)
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge")
    parser.add_argument("--model", type=str, default="gemma4:31b-cloud", help="LLM for generic rules")
    parser.add_argument("--judge-model", type=str, default=None, help="LLM judge (default: same as --model)")
    parser.add_argument("--provider", type=str, default="ollama")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434/v1")
    parser.add_argument("--api-key", type=str, default="ollama")
    parser.add_argument("--bert-model", type=str, default="Jzuluaga/bert-base-ner-atc-en-atco2-1h")
    parser.add_argument("--skip-bert", action="store_true", help="Skip BERT NER (regex parse only, faster)")
    parser.add_argument("--skip-generic", action="store_true", help="Skip generic LLM rules (native+compiled only)")
    parser.add_argument("--no-llm-batch", action="store_true", help="Disable RuleFilter LLM batch layer")
    parser.add_argument("--filter-top-k", type=int, default=10)
    parser.add_argument("--filter-timeout", type=float, default=30.0)
    parser.add_argument("--generic-timeout", type=float, default=300.0, help="Generic rule eval timeout (s)")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--tc-ids",
        type=str,
        default=None,
        help="Comma-separated test case IDs to run (default: all)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete partial_results.json before running",
    )
    args = parser.parse_args()

    cfg = E7Config(
        bert=BERTConfig(model_name=args.bert_model),
        llm=LLMConfig(
            model_name=args.model,
            provider=args.provider,
            base_url=args.base_url,
            api_key=args.api_key,
            max_retries=args.max_retries,
            timeout=int(args.generic_timeout),
        ),
        judge_model=args.judge_model or args.model,
        filter=FilterConfig(
            top_k=args.filter_top_k,
            timeout_seconds=args.filter_timeout,
            verbose=not args.quiet,
        ),
        generic_eval_timeout_s=args.generic_timeout,
        generic_max_retries=args.max_retries,
        verbose=not args.quiet,
        skip_bert=args.skip_bert,
        skip_generic=args.skip_generic,
    )

    if args.no_llm_batch:
        cfg.filter.use_llm_batch = False

    cfg.resolve_paths(PROJECT_ROOT)

    if args.ground_truth_dir:
        cfg.ground_truth_dir = Path(args.ground_truth_dir)
    if args.results_dir:
        cfg.results_dir = Path(args.results_dir)
        cfg.figures_dir = cfg.results_dir / "figures"
    if args.compiled_rules_dir:
        cfg.compiled_rules_dir = Path(args.compiled_rules_dir)
    if args.rules_json:
        cfg.rules_json_path = Path(args.rules_json)

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        partial_path = cfg.results_dir / cfg.partial_file
        if partial_path.exists():
            partial_path.unlink()
            print(f"  Cleared partial results: {partial_path}")

    tc_ids = [x.strip() for x in args.tc_ids.split(",") if x.strip()] if args.tc_ids else None

    print("E7 Integrated Pipeline Benchmark")
    print(f"  Ground truth: {cfg.ground_truth_dir}")
    print(f"  Compiled rules: {cfg.compiled_rules_dir}")
    print(f"  Generic rules: {cfg.rules_json_path}")
    print(f"  LLM: {args.model} ({args.provider})")
    print(f"  BERT: {args.bert_model}" + (" [SKIP]" if args.skip_bert else ""))
    print(f"  Generic rules: {'SKIP' if args.skip_generic else 'ON'}")
    print(f"  RuleFilter LLM batch: {'OFF' if args.no_llm_batch else 'ON'}")
    print(f"  RuleFilter: top_k={args.filter_top_k}, timeout={args.filter_timeout}s")
    print(f"  Generic eval timeout: {args.generic_timeout}s")
    if args.no_judge:
        print("  Judge: disabled")
    else:
        print(f"  Judge: {cfg.judge_model}")
    if tc_ids:
        print(f"  TC subset: {', '.join(tc_ids)}")
    print()

    judge = None
    if not args.no_judge:
        judge = PipelineJudge(
            LLMConfig(
                model_name=cfg.judge_model,
                provider=args.provider,
                base_url=args.base_url,
                api_key=args.api_key,
                max_retries=args.max_retries,
                timeout=120,
            )
        )

    results = run_benchmark(cfg, judge=judge, tc_ids=tc_ids)
    report_info = generate_report(results, cfg, save_figures=True)

    agg = results.aggregate_accuracy
    print(f"Report: {report_info['results']}")
    print(f"Summary: {report_info['summary']}")
    print(f"Figures: {len(report_info['figures'])} generated")
    print(
        f"Aggregate accuracy: P={agg.get('precision', 0):.3f} "
        f"R={agg.get('recall', 0):.3f} F1={agg.get('f1', 0):.3f} "
        f"SeverityAcc={agg.get('severity_accuracy', 0):.3f}"
    )

    if results.judge_scores:
        avg_judge = sum(results.judge_scores.values()) / len(results.judge_scores)
        print(f"Avg judge score: {avg_judge:.3f}")

    e2e = results.latency.get("e2e", {})
    if e2e:
        print(f"E2E latency: avg={e2e.get('avg_ms', 0):.0f}ms p95={e2e.get('p95_ms', 0):.0f}ms")


if __name__ == "__main__":
    main()
