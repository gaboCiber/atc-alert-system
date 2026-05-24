#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import E3Config, EvalConfig
from loader import ExperimentData
from evaluator import run_evaluation
from report import generate_report


def main():
    parser = argparse.ArgumentParser(
        description="E3: ASR Transcription Comparison Between Models and Ground Truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                                                  # default paths
  python run.py --gt-dir /path/to/gt --transcriptions-dir /path/to/transcriptions
  python run.py --no-atc-normalizer                              # skip ATC normalization
        """,
    )

    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--gt-dir", type=str, default=None)
    parser.add_argument("--transcriptions-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-atc-normalizer", action="store_true", help="Skip ATC text normalization")
    parser.add_argument("--no-jiwer", action="store_true", help="Use basic WER instead of jiwer")

    args = parser.parse_args()

    cfg = E3Config.from_dirs(
        base_dir=args.base_dir,
        ground_truth_dir=args.gt_dir,
        transcriptions_dir=args.transcriptions_dir,
        output_dir=args.output,
    )

    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = EvalConfig(
        use_atc_normalizer=not args.no_atc_normalizer,
        use_jiwer=not args.no_jiwer,
    )

    print("=" * 60)
    print("E3: ASR Transcription Comparison Experiment")
    print("=" * 60)
    print(f"  Ground truth:    {cfg.ground_truth_dir}")
    print(f"  Transcriptions:  {cfg.transcriptions_dir}")
    print(f"  Output:          {cfg.results_dir}")
    print(f"  ATC Normalizer:  {'enabled' if eval_cfg.use_atc_normalizer else 'disabled'}")
    print(f"  Jiwer:           {'enabled' if eval_cfg.use_jiwer else 'disabled'}")
    print()

    print("Loading experiment data...")
    try:
        data = ExperimentData.from_config(cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"  Found {len(data.model_names)} model(s):")
    for mn in data.model_names:
        print(f"    - {mn}")
    print(f"  Common samples:  {len(data.common_ids)}")
    print()

    print("Running evaluation...")
    results = run_evaluation(data, eval_cfg)
    print(f"  Evaluated {len(data.common_ids)} samples across {len(data.model_names)} models")
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
    print(f"  Results: {report['results']}")
    print(f"  Summary: {report['summary']}")
    if not args.no_figures:
        print("  Figures:")
        for name, path in report["figures"].items():
            print(f"    - {name}.png")

    print()
    print("MODEL RANKING (by WER, lower is better):")
    for entry in report.get("ranking", []):
        print(f"  #{entry['rank']}: {entry['model']}")
        print(f"       WER={entry['wer']:.2%} | MER={entry['mer']:.2%} | WIL={entry['wil']:.2%} | WIP={entry['wip']:.2%}")
        print(f"       Subs={entry['substitutions']} | Ins={entry['insertions']} | Dels={entry['deletions']}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
