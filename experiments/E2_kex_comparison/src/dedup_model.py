#!/usr/bin/env python3
"""
Standalone script: detect semantic duplicates within a single model's output.

Useful for pseudo-GT creation: given a model directory, cluster semantically
equivalent artifacts across all pages so you can manually merge them.

Usage:
  uv run scripts/dedup_model.py --model-dir models/gemini-3-flash-preview
  uv run scripts/dedup_model.py --model-dir models/gemini-3-flash-preview --batch-size 10 --threshold 0.80
  uv run scripts/dedup_model.py --model-dir models/gemini-3-flash-preview --judge-model llama3.2 --no-save
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import JudgeConfig, DedupConfig
from src.loader import load_model_pages
from src.llm_judge import LLMJudge
from src.dedup import analyze_model, format_card


def main():
    parser = argparse.ArgumentParser(description="Detect semantic duplicates within a model's KEX output")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory (default: same as model dir)")
    parser.add_argument("--no-save", action="store_true", help="Print to stdout only, don't save files")
    parser.add_argument("--batch-size", type=int, default=10, help="LLM batch size (default: 10)")
    parser.add_argument("--threshold", type=float, default=0.80, help="Similarity threshold for dedup (default: 0.80)")
    parser.add_argument("--judge-model", type=str, default="llama3.2", help="LLM model for judging")
    parser.add_argument("--judge-provider", type=str, default="openai", choices=["openai", "ollama", "gemini", "anthropic"])
    parser.add_argument("--judge-base-url", type=str, default="http://localhost:11434/v1")
    parser.add_argument("--judge-api-key", type=str, default="ollama")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory for checkpoints (default: output dir)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: model directory not found: {model_dir}")
        sys.exit(1)

    provider = args.judge_provider
    if provider == "ollama":
        provider = "openai"

    judge_cfg = JudgeConfig(
        model_name=args.judge_model,
        provider=provider,
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
        enabled=True,
    )

    dedup_cfg = DedupConfig(
        enabled=True,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )

    # Generate config hash for checkpoint validation
    dedup_config_hash = hashlib.sha256(
        json.dumps({
            "judge": {
                "model_name": judge_cfg.model_name,
                "provider": judge_cfg.provider,
                "base_url": judge_cfg.base_url,
                "api_key": judge_cfg.api_key,
            },
            "dedup": {
                "batch_size": dedup_cfg.batch_size,
                "threshold": dedup_cfg.threshold,
            },
        }, sort_keys=True).encode()
    ).hexdigest()

    output_dir = Path(args.output) if args.output else model_dir
    results_dir = Path(args.results_dir) if args.results_dir else output_dir

    print(f"Loading model: {model_dir.name}")
    model_result = load_model_pages(model_dir)
    if not model_result.available_pages:
        print("ERROR: no pages loaded from model directory")
        sys.exit(1)
    print(f"  Pages: {model_result.available_pages}")
    print(f"  Judge: {judge_cfg.model_name} ({provider})")
    print(f"  Batch: {dedup_cfg.batch_size}, threshold: {dedup_cfg.threshold}")
    if results_dir:
        print(f"  Checkpoints: {results_dir / 'checkpoints' / f'dedup_intra_{model_dir.name}.json'}")
    print()

    judge = LLMJudge(judge_cfg)
    report = analyze_model(
        model_result=model_result,
        judge=judge,
        batch_size=dedup_cfg.batch_size,
        threshold=dedup_cfg.threshold,
        results_dir=results_dir,
        config_hash=dedup_config_hash,
    )

    card = format_card(report)
    print(card)

    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / f"dedup_{model_dir.name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        txt_path = output_dir / f"dedup_{model_dir.name}_card.txt"
        txt_path.write_text(card, encoding="utf-8")

        print()
        print(f"  JSON report:     {json_path}")
        print(f"  TXT card:        {txt_path}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
