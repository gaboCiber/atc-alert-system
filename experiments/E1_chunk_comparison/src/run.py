#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import E1Config
from loader import ExperimentData, load_sentence_gt
from evaluator import run_evaluation
from report import generate_report


def main():
    parser = argparse.ArgumentParser(
        description="E1: Chunk comparison between models and Ground Truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                                                  # default paths
  python run.py --base-dir ./experiments/E1_chunk_comparison
  python run.py --gt-dir /path/to/gt --models-dir /path/to/models
  python run.py --output /path/to/output --no-figures
        """,
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base experiment directory (default: this script's parent)",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default=None,
        help="Ground truth directory (default: <base>/ground_truth/)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Models directory (default: <base>/models/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: <base>/results/)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI (default: 150)",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="seaborn-v0.8-darkgrid",
        help="Matplotlib style (default: seaborn-v0.8-darkgrid)",
    )

    args = parser.parse_args()

    cfg = E1Config.from_dirs(
        base_dir=args.base_dir,
        ground_truth_dir=args.gt_dir,
        models_dir=args.models_dir,
        output_dir=args.output,
    )

    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("E1: Chunk Comparison Experiment")
    print("=" * 60)
    print(f"  Ground truth: {cfg.ground_truth_dir}")
    print(f"  Models:      {cfg.models_dir}")
    print(f"  Output:      {cfg.results_dir}")
    print(f"  Figures:     {'enabled' if not args.no_figures else 'disabled'}")
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

    print("Running evaluation...")
    try:
        sentences_gt = load_sentence_gt(cfg.sentences_gt_path)
        print(f"  Using sentence GT: {cfg.sentences_gt_path}")
    except FileNotFoundError:
        sentences_gt = None
        print("  No sentence GT found, falling back to NLTK for BoundaryIntegrity")
    results = run_evaluation(data, sentences_gt=sentences_gt)
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
        print(f"       Score={entry['overall_score']:.3f}  BoundaryF1={entry['boundary_f1_mean']:.3f}  ContentF1={entry['matched_content_f1_mean']:.3f}  ChunkAcc={entry['chunk_count_accuracy_mean']:.3f}  BoundaryInt={entry['boundary_integrity_mean']:.3f}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()