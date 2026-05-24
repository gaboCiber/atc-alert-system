from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np

from config import E1Config, MetricConfig
from loader import ExperimentData, ModelResult
from metrics import PageMetrics, compute_page_metrics


@dataclass
class ModelPageMetrics:
    page: int
    model: str
    metrics: PageMetrics


@dataclass
class ModelSummaryMetrics:
    model: str
    page_metrics: List[PageMetrics] = field(default_factory=list)

    chunk_count_accuracy_mean: float = 0.0
    chunk_count_accuracy_std: float = 0.0
    boundary_f1_mean: float = 0.0
    boundary_f1_std: float = 0.0
    boundary_precision_mean: float = 0.0
    boundary_recall_mean: float = 0.0
    boundary_avg_error_mean: float = 0.0

    char_f1_mean: float = 0.0
    char_f1_std: float = 0.0
    word_f1_mean: float = 0.0
    word_f1_std: float = 0.0
    rouge_l_mean: float = 0.0
    token_overlap_mean: float = 0.0
    fuzzy_match_mean: float = 0.0

    overall_score: float = 0.0


def _compute_summary(model_name: str, page_metrics_list: List[PageMetrics]) -> ModelSummaryMetrics:
    s = ModelSummaryMetrics(model=model_name, page_metrics=page_metrics_list)

    cca = [pm.structural.chunk_count_accuracy for pm in page_metrics_list]
    s.chunk_count_accuracy_mean = np.mean(cca)
    s.chunk_count_accuracy_std = np.std(cca)

    bf1 = [pm.structural.boundary_f1 for pm in page_metrics_list]
    s.boundary_f1_mean = np.mean(bf1)
    s.boundary_f1_std = np.std(bf1)
    s.boundary_precision_mean = np.mean([pm.structural.boundary_precision for pm in page_metrics_list])
    s.boundary_recall_mean = np.mean([pm.structural.boundary_recall for pm in page_metrics_list])
    s.boundary_avg_error_mean = np.mean([pm.structural.boundary_avg_error for pm in page_metrics_list])

    s.char_f1_mean = np.mean([pm.content.char_f1 for pm in page_metrics_list])
    s.char_f1_std = np.std([pm.content.char_f1 for pm in page_metrics_list])
    s.word_f1_mean = np.mean([pm.content.word_f1 for pm in page_metrics_list])
    s.word_f1_std = np.std([pm.content.word_f1 for pm in page_metrics_list])
    s.rouge_l_mean = np.mean([pm.content.rouge_l for pm in page_metrics_list])
    s.token_overlap_mean = np.mean([pm.content.token_overlap_ratio for pm in page_metrics_list])
    s.fuzzy_match_mean = np.mean([pm.content.fuzzy_match_ratio for pm in page_metrics_list])

    weights = {
        "chunk_count_accuracy": 0.15,
        "boundary_f1": 0.20,
        "char_f1": 0.25,
        "word_f1": 0.20,
        "rouge_l": 0.20,
    }
    score = (
        weights["chunk_count_accuracy"] * s.chunk_count_accuracy_mean
        + weights["boundary_f1"] * s.boundary_f1_mean
        + weights["char_f1"] * s.char_f1_mean
        + weights["word_f1"] * s.word_f1_mean
        + weights["rouge_l"] * s.rouge_l_mean
    )
    s.overall_score = score

    return s


@dataclass
class EvaluationResults:
    model_names: List[str]
    pages: List[int]
    page_results: Dict[str, List[PageMetrics]]
    summaries: Dict[str, ModelSummaryMetrics]

    def to_dict(self) -> dict:
        page_results_serializable = {}
        for model, pms in self.page_results.items():
            page_results_serializable[model] = [
                {
                    "page": pm.page,
                    "structural": {
                        "chunk_count": pm.structural.chunk_count,
                        "gt_chunk_count": pm.structural.gt_chunk_count,
                        "chunk_count_error": pm.structural.chunk_count_error,
                        "chunk_count_accuracy": pm.structural.chunk_count_accuracy,
                        "boundary_f1": pm.structural.boundary_f1,
                        "boundary_precision": pm.structural.boundary_precision,
                        "boundary_recall": pm.structural.boundary_recall,
                        "boundary_avg_error": pm.structural.boundary_avg_error,
                    },
                    "content": {
                        "char_f1": pm.content.char_f1,
                        "char_precision": pm.content.char_precision,
                        "char_recall": pm.content.char_recall,
                        "word_f1": pm.content.word_f1,
                        "word_precision": pm.content.word_precision,
                        "word_recall": pm.content.word_recall,
                        "rouge_l": pm.content.rouge_l,
                        "token_overlap_ratio": pm.content.token_overlap_ratio,
                        "fuzzy_match_ratio": pm.content.fuzzy_match_ratio,
                    },
                }
                for pm in pms
            ]

        summaries_serializable = {}
        for model, sm in self.summaries.items():
            summaries_serializable[model] = {
                "model": sm.model,
                "overall_score": sm.overall_score,
                "structural": {
                    "chunk_count_accuracy_mean": sm.chunk_count_accuracy_mean,
                    "chunk_count_accuracy_std": sm.chunk_count_accuracy_std,
                    "boundary_f1_mean": sm.boundary_f1_mean,
                    "boundary_f1_std": sm.boundary_f1_std,
                    "boundary_precision_mean": sm.boundary_precision_mean,
                    "boundary_recall_mean": sm.boundary_recall_mean,
                    "boundary_avg_error_mean": sm.boundary_avg_error_mean,
                },
                "content": {
                    "char_f1_mean": sm.char_f1_mean,
                    "char_f1_std": sm.char_f1_std,
                    "word_f1_mean": sm.word_f1_mean,
                    "word_f1_std": sm.word_f1_std,
                    "rouge_l_mean": sm.rouge_l_mean,
                    "token_overlap_mean": sm.token_overlap_mean,
                    "fuzzy_match_mean": sm.fuzzy_match_mean,
                },
            }

        return {
            "model_names": self.model_names,
            "pages": self.pages,
            "page_results": page_results_serializable,
            "summaries": summaries_serializable,
        }


def run_evaluation(data: ExperimentData) -> EvaluationResults:
    page_results: Dict[str, List[PageMetrics]] = {model: [] for model in data.model_names}

    for page in data.pages:
        gt = data.ground_truth[page]
        for model_name in data.model_names:
            model_result = data.model_results[model_name]
            if page not in model_result.pages:
                continue
            pred = model_result.pages[page]
            pm = compute_page_metrics(page, pred, gt)
            page_results[model_name].append(pm)

    summaries: Dict[str, ModelSummaryMetrics] = {}
    for model_name in data.model_names:
        summaries[model_name] = _compute_summary(model_name, page_results[model_name])

    return EvaluationResults(
        model_names=data.model_names,
        pages=data.pages,
        page_results=page_results,
        summaries=summaries,
    )