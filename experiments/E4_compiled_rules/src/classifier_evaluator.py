from dataclasses import dataclass, field
from typing import Dict, List

from loader import ModelCompilationResult, ClassificationGT


@dataclass
class ClassificationResult:
    rule_id: str
    model_name: str
    model_decision: bool
    gt_decision: bool
    correct: bool
    confidence: float = 0.0
    reason: str = ""


@dataclass
class ClassificationMetrics:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    total_rules: int = 0
    results: List[ClassificationResult] = field(default_factory=list)


def evaluate_classification(
    model_result: ModelCompilationResult,
    gt_classification: Dict[str, ClassificationGT],
) -> ClassificationMetrics:
    cm = ClassificationMetrics(model_name=model_result.model_name, accuracy=0.0, precision=0.0, recall=0.0)
    results = []

    all_rule_ids = set(model_result.rule_ids) | set(model_result.failed_rules.keys())

    for rule_id in all_rule_ids:
        gt = gt_classification.get(rule_id)
        if gt is None:
            continue

        if rule_id in model_result.compiled_rules:
            model_compilable = True
            conf = 1.0
            reason = "compiled"
        elif rule_id in model_result.failed_rules:
            status = model_result.failed_rules[rule_id]
            model_compilable = False
            conf = 1.0
            reason = f"failed: {status}"
        else:
            continue

        correct = model_compilable == gt.is_compilable

        results.append(ClassificationResult(
            rule_id=rule_id,
            model_name=model_result.model_name,
            model_decision=model_compilable,
            gt_decision=gt.is_compilable,
            correct=correct,
            confidence=conf,
            reason=reason,
        ))

        if model_compilable and gt.is_compilable:
            cm.true_positives += 1
        elif model_compilable and not gt.is_compilable:
            cm.false_positives += 1
        elif not model_compilable and gt.is_compilable:
            cm.false_negatives += 1
        else:
            cm.true_negatives += 1

    cm.results = results
    cm.total_rules = len(results)

    if cm.total_rules > 0:
        cm.accuracy = sum(1 for r in results if r.correct) / cm.total_rules

    tp = cm.true_positives
    fp = cm.false_positives
    fn = cm.false_negatives

    cm.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    cm.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return cm


def evaluate_all_classifications(
    model_results: Dict[str, ModelCompilationResult],
    gt_classification: Dict[str, ClassificationGT],
) -> Dict[str, ClassificationMetrics]:
    return {
        model_name: evaluate_classification(result, gt_classification)
        for model_name, result in model_results.items()
    }