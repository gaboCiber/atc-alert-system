#!/usr/bin/env python3
import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import E3Config, EvalConfig
from loader import ExperimentData
from evaluator import run_evaluation, EvaluationResults
from report import generate_report, generate_combined_figures


def run_single_dataset(
    cfg: E3Config,
    eval_cfg: EvalConfig,
    dataset_name: str = "",
    save_figures: bool = True,
) -> tuple:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading experiment data...")
    try:
        data = ExperimentData.from_config(cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"  SKIP: {e}")
        return None, None

    print(f"  Found {len(data.model_names)} model(s)")
    print(f"  Common samples: {len(data.common_ids)}")
    print()

    print("Running evaluation...")
    results = run_evaluation(data, eval_cfg)
    print(f"  Evaluated {len(data.common_ids)} samples across {len(data.model_names)} models")
    print()

    print("Generating report...")
    suffix = f"_{dataset_name}" if dataset_name else ""
    report = generate_report(
        results,
        cfg,
        save_figures=save_figures,
        dataset_suffix=suffix,
    )

    print("MODEL RANKING (by WER, lower is better):")
    for entry in report.get("ranking", []):
        print(f"  #{entry['rank']}: {entry['model']}")
        print(f"       WER={entry['wer']:.2%} | MER={entry['mer']:.2%} | WIL={entry['wil']:.2%} | WIP={entry['wip']:.2%}")
        print(f"       Subs={entry['substitutions']} | Ins={entry['insertions']} | Dels={entry['deletions']}")
    print()

    return report, results


def main():
    parser = argparse.ArgumentParser(
        description="E3: ASR Transcription Comparison Between Models and Ground Truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --gt-file gt.csv --transcriptions-file trans.csv
  python run.py --auto-discover --output results_all
  python run.py --datasets atco2,jacktol_test --output results_all
  python run.py --no-atc-normalizer
        """,
    )

    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--gt-dir", type=str, default=None)
    parser.add_argument("--transcriptions-dir", type=str, default=None)
    parser.add_argument("--gt-file", type=str, default=None)
    parser.add_argument("--transcriptions-file", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-atc-normalizer", action="store_true")
    parser.add_argument("--no-jiwer", action="store_true")
    parser.add_argument("--auto-discover", action="store_true", help="Auto-discover dataset pairs")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names to run")

    args = parser.parse_args()

    eval_cfg = EvalConfig(
        use_atc_normalizer=not args.no_atc_normalizer,
        use_jiwer=not args.no_jiwer,
    )

    if args.auto_discover or args.datasets:
        base = Path(args.base_dir) if args.base_dir else Path(__file__).parent.parent
        gt_dir = Path(args.gt_dir) if args.gt_dir else base / "ground_truth"
        trans_dir = Path(args.transcriptions_dir) if args.transcriptions_dir else base / "transcriptions"
        output_base = Path(args.output) if args.output else base / "results_all"

        print("=" * 60)
        print("E3: Multi-Dataset ASR Comparison")
        print("=" * 60)
        print(f"  Ground truth dir:  {gt_dir}")
        print(f"  Transcriptions dir: {trans_dir}")
        print(f"  Output:             {output_base}")
        print()

        pairs = E3Config.discover_dataset_pairs(gt_dir, trans_dir)

        if args.datasets:
            allowed = set(args.datasets.split(","))
            pairs = [(n, gt, t) for n, gt, t in pairs if n in allowed]

        if not pairs:
            print("ERROR: No matching dataset pairs found")
            sys.exit(1)

        print(f"Found {len(pairs)} dataset(s): {', '.join(n for n, _, _ in pairs)}")
        print()

        all_eval_results = {}
        for dataset_name, gt_path, trans_path in pairs:
            print("=" * 60)
            print(f"  DATASET: {dataset_name}")
            print("=" * 60)

            cfg = E3Config.from_dirs(
                base_dir=args.base_dir,
                ground_truth_dir=str(gt_dir),
                transcriptions_dir=str(trans_dir),
                ground_truth_file=str(gt_path),
                transcriptions_file=str(trans_path),
                output_dir=str(output_base / dataset_name),
                dataset_name=dataset_name,
            )

            report, results = run_single_dataset(
                cfg, eval_cfg,
                dataset_name=dataset_name,
                save_figures=not args.no_figures,
            )
            if results:
                all_eval_results[dataset_name] = results

        if len(all_eval_results) >= 2:
            print("=" * 60)
            print("Generating combined cross-dataset figures...")
            print("=" * 60)
            combined = generate_combined_figures(
                all_eval_results,
                output_base,
                save_figures=not args.no_figures,
            )
            for name, path in combined["figures"].items():
                print(f"  - {name}.png")

        print()
        print("=" * 60)
        print("MULTI-DATASET COMPLETE")
        print("=" * 60)

    else:
        cfg = E3Config.from_dirs(
            base_dir=args.base_dir,
            ground_truth_dir=args.gt_dir,
            transcriptions_dir=args.transcriptions_dir,
            ground_truth_file=args.gt_file,
            transcriptions_file=args.transcriptions_file,
            output_dir=args.output,
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

        run_single_dataset(cfg, eval_cfg, save_figures=not args.no_figures)

    print("Done.")


if __name__ == "__main__":
    main()
