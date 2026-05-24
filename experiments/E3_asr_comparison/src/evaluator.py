import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from ASR.evaluation.evaluator import ASREvaluator, ASREvaluationResult

from config import E3Config, EvalConfig
from loader import ExperimentData


@dataclass
class ModelSummaryMetrics:
    model: str
    results: List[ASREvaluationResult] = field(default_factory=list)

    average_wer: float = 0.0
    average_mer: float = 0.0
    average_wil: float = 0.0
    average_wip: float = 0.0
    total_ref_words: int = 0
    total_hyp_words: int = 0
    num_samples: int = 0
    total_substitutions: int = 0
    total_insertions: int = 0
    total_deletions: int = 0
    total_hits: int = 0

    per_sample_wer: List[float] = field(default_factory=list)
    per_sample_cer: List[float] = field(default_factory=list)


@dataclass
class EvaluationResults:
    model_names: List[str]
    common_ids: List[str]
    model_results: Dict[str, List[ASREvaluationResult]] = field(default_factory=dict)
    summaries: Dict[str, ModelSummaryMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict:
        model_data = {}
        for model, results in self.model_results.items():
            sm = self.summaries.get(model)
            model_data[model] = {
                "summary": {
                    "average_wer": sm.average_wer if sm else 0.0,
                    "average_mer": sm.average_mer if sm else 0.0,
                    "average_wil": sm.average_wil if sm else 0.0,
                    "average_wip": sm.average_wip if sm else 0.0,
                    "total_ref_words": sm.total_ref_words if sm else 0,
                    "total_hyp_words": sm.total_hyp_words if sm else 0,
                    "num_samples": sm.num_samples if sm else 0,
                    "total_substitutions": sm.total_substitutions if sm else 0,
                    "total_insertions": sm.total_insertions if sm else 0,
                    "total_deletions": sm.total_deletions if sm else 0,
                    "total_hits": sm.total_hits if sm else 0,
                },
                "per_sample": [
                    {
                        "sample_id": r.timestamp,
                        "wer": r.wer,
                        "mer": r.mer,
                        "wil": r.wil,
                        "wip": r.wip,
                        "cer": r.cer,
                        "num_ref_words": r.num_ref_words,
                        "num_hyp_words": r.num_hyp_words,
                        "errors": r.errors,
                    }
                    for r in results
                ],
            }

        return {
            "model_names": self.model_names,
            "num_samples": len(self.common_ids),
            "models": model_data,
        }


def run_evaluation(
    data: ExperimentData,
    eval_cfg: EvalConfig,
) -> EvaluationResults:
    evaluator = ASREvaluator(
        use_jiwer=eval_cfg.use_jiwer,
        normalize_words=True,
        use_atc_normalizer=eval_cfg.use_atc_normalizer,
    )

    model_results: Dict[str, List[ASREvaluationResult]] = {}

    for model_name in data.model_names:
        model_transcriptions = data.get_model_transcriptions(model_name)
        results = []

        for sample_id in data.common_ids:
            ref = data.ground_truth.get(sample_id, "")
            hyp = model_transcriptions.get(sample_id, "")

            metrics = evaluator.calculate_wer(ref, hyp, detailed=eval_cfg.detailed)

            result = ASREvaluationResult(
                model_name=model_name,
                timestamp=sample_id,
                reference=ref,
                hypothesis=hyp,
                wer=metrics["wer"],
                mer=metrics.get("mer"),
                wil=metrics.get("wil"),
                wip=metrics.get("wip"),
                num_ref_words=metrics["num_ref_words"],
                num_hyp_words=metrics["num_hyp_words"],
                errors=metrics.get("errors"),
            )
            results.append(result)

        model_results[model_name] = results

    summaries: Dict[str, ModelSummaryMetrics] = {}
    for model_name in data.model_names:
        results = model_results[model_name]
        sm = ModelSummaryMetrics(model=model_name, results=results)

        if results:
            total_ref = sum(r.num_ref_words for r in results)
            total_hyp = sum(r.num_hyp_words for r in results)
            total_errors = sum(r.wer * r.num_ref_words for r in results)
            sm.average_wer = total_errors / total_ref if total_ref > 0 else 0.0
            sm.total_ref_words = total_ref
            sm.total_hyp_words = total_hyp
            sm.num_samples = len(results)

            mers = [r.mer for r in results if r.mer is not None]
            sm.average_mer = sum(mers) / len(mers) if mers else 0.0

            wils = [r.wil for r in results if r.wil is not None]
            sm.average_wil = sum(wils) / len(wils) if wils else 0.0

            wips = [r.wip for r in results if r.wip is not None]
            sm.average_wip = sum(wips) / len(wips) if wips else 0.0

            all_errors = [r.errors for r in results if r.errors]
            if all_errors:
                sm.total_substitutions = sum(e["substitutions"] for e in all_errors)
                sm.total_insertions = sum(e["insertions"] for e in all_errors)
                sm.total_deletions = sum(e["deletions"] for e in all_errors)
                sm.total_hits = sum(e["hits"] for e in all_errors)

            sm.per_sample_wer = [r.wer for r in results]
            sm.per_sample_cer = [r.cer for r in results if r.cer is not None]

        summaries[model_name] = sm

    return EvaluationResults(
        model_names=data.model_names,
        common_ids=data.common_ids,
        model_results=model_results,
        summaries=summaries,
    )
