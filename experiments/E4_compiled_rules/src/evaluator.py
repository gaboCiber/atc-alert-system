from dataclasses import dataclass, field
from typing import Dict, List, Optional
from tqdm import tqdm

from config import E4Config, JudgeConfig, MetricConfig
from loader import ExperimentData
from classifier_evaluator import evaluate_all_classifications, ClassificationMetrics
from validator import validate_rules
from executor import evaluate_compiled_rules, ExecutionMetrics
from semantic_judge import SemanticJudge


@dataclass
class SemanticMetrics:
    model_name: str
    scores: Dict[str, float] = field(default_factory=dict)
    mean_score: float = 0.0
    count: int = 0


@dataclass
class ModelSummaryMetrics:
    model: str
    classification_metrics: Optional[ClassificationMetrics] = None
    validation_pass_rate: float = 0.0
    validation_total: int = 0
    execution_metrics: Optional[ExecutionMetrics] = None
    semantic_metrics: Optional[SemanticMetrics] = None
    overall_score: float = 0.0


@dataclass
class EvaluationResults:
    model_names: List[str]
    classification_results: Dict[str, ClassificationMetrics] = field(default_factory=dict)
    validation_results: Dict[str, Dict] = field(default_factory=dict)
    execution_results: Dict[str, ExecutionMetrics] = field(default_factory=dict)
    semantic_results: Dict[str, SemanticMetrics] = field(default_factory=dict)
    summaries: Dict[str, ModelSummaryMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_names": self.model_names,
            "classification": {
                m: {
                    "accuracy": r.accuracy,
                    "precision": r.precision,
                    "recall": r.recall,
                    "total_rules": r.total_rules,
                    "tp": r.true_positives,
                    "fp": r.false_positives,
                    "tn": r.true_negatives,
                    "fn": r.false_negatives,
                }
                for m, r in self.classification_results.items()
            },
            "validation": {
                m: {
                    "pass_rate": sum(1 for v in vals.values() if v.passed) / len(vals) if vals else 0.0,
                    "total": len(vals),
                }
                for m, vals in self.validation_results.items()
            },
            "execution": {
                m: {
                    "match_rate": r.match_rate,
                    "successful": r.successful_executions,
                    "failed": r.failed_executions,
                }
                for m, r in self.execution_results.items()
            },
            "semantic": {
                m: {
                    "mean_score": r.mean_score,
                    "count": r.count,
                }
                for m, r in self.semantic_results.items()
            },
            "summaries": {
                m: {
                    "overall_score": s.overall_score,
                    "classification_accuracy": s.classification_metrics.accuracy if s.classification_metrics else 0.0,
                    "validation_pass_rate": s.validation_pass_rate,
                    "execution_match_rate": s.execution_metrics.match_rate if s.execution_metrics else 0.0,
                    "semantic_score": s.semantic_metrics.mean_score if s.semantic_metrics else 0.0,
                }
                for m, s in self.summaries.items()
            },
        }


def run_evaluation(
    data: ExperimentData,
    judge: SemanticJudge,
    metric_cfg: MetricConfig,
) -> EvaluationResults:
    classification_results = evaluate_all_classifications(
        data.model_results, data.ground_truth_classification
    )

    validation_results = {}
    for model_name, model_result in data.model_results.items():
        compiled_code = {rid: cr.compiled_code for rid, cr in model_result.compiled_rules.items()}
        validation_results[model_name] = validate_rules(
            list(model_result.compiled_rules.keys()), compiled_code, model_name
        )

    execution_results = {}
    for model_name, model_result in data.model_results.items():
        if model_result.compiled_rules and data.test_traffic_states:
            execution_results[model_name] = evaluate_compiled_rules(
                model_result.compiled_rules,
                data.test_traffic_states,
                model_name,
            )

    semantic_results = {}
    if judge.config.enabled and data.reference_code:
        for model_name, model_result in data.model_results.items():
            compiled_code = {rid: cr.compiled_code for rid, cr in model_result.compiled_rules.items() if cr.compiled_code}
            if not compiled_code:
                continue

            sm = SemanticMetrics(model_name=model_name, scores={})
            for rule_id in tqdm(compiled_code.keys(), desc=f"Judging {model_name}"):
                gt_code = data.reference_code.get(rule_id, "")
                gen_code = compiled_code[rule_id]
                if gt_code and gen_code:
                    result = judge.judge(rule_id, gt_code, gen_code)
                    if result:
                        sm.scores[rule_id] = result.similarity_score

            if sm.scores:
                sm.mean_score = sum(sm.scores.values()) / len(sm.scores)
                sm.count = len(sm.scores)
            semantic_results[model_name] = sm

    summaries = {}
    for model_name in data.model_names:
        sm = ModelSummaryMetrics(model=model_name)
        sm.classification_metrics = classification_results.get(model_name)

        val_res = validation_results.get(model_name, {})
        if val_res:
            sm.validation_total = len(val_res)
            sm.validation_pass_rate = sum(1 for v in val_res.values() if v.passed) / len(val_res)

        sm.execution_metrics = execution_results.get(model_name)
        sm.semantic_metrics = semantic_results.get(model_name)

        cl_acc = sm.classification_metrics.accuracy if sm.classification_metrics else 0.0
        val_rate = sm.validation_pass_rate
        exec_rate = sm.execution_metrics.match_rate if sm.execution_metrics else 0.0
        sem_score = sm.semantic_metrics.mean_score if sm.semantic_metrics else 0.0

        sm.overall_score = (
            metric_cfg.classification_weight * cl_acc
            + metric_cfg.validation_weight * val_rate
            + metric_cfg.execution_weight * exec_rate
            + metric_cfg.semantic_weight * sem_score
        )

        summaries[model_name] = sm

    return EvaluationResults(
        model_names=data.model_names,
        classification_results=classification_results,
        validation_results=validation_results,
        execution_results=execution_results,
        semantic_results=semantic_results,
        summaries=summaries,
    )