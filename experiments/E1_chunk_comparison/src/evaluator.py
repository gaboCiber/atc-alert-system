from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config import E1Config
from loader import ExperimentData, ModelResult
from metrics import PageMetrics, compute_boundary_integrity_document, compute_page_metrics


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
    chunk_count_error_mean: float = 0.0
    chunk_count_error_std: float = 0.0
    boundary_f1_mean: float = 0.0
    boundary_f1_std: float = 0.0
    boundary_precision_mean: float = 0.0
    boundary_recall_mean: float = 0.0
    boundary_avg_error_mean: float = 0.0

    matched_content_precision_mean: float = 0.0
    matched_content_recall_mean: float = 0.0
    matched_content_f1_mean: float = 0.0
    matched_content_f1_std: float = 0.0

    boundary_integrity_mean: float = 0.0
    boundary_integrity_std: float = 0.0
    boundary_integrity_detail: Dict = field(default_factory=dict)

    overall_score: float = 0.0


def _compute_summary(
    model_name: str,
    page_metrics_list: List[PageMetrics],
    model_result: ModelResult,
    sentences_gt: Optional[List[str]] = None,
) -> ModelSummaryMetrics:
    s = ModelSummaryMetrics(model=model_name, page_metrics=page_metrics_list)

    cca = [pm.structural.chunk_count_accuracy for pm in page_metrics_list]
    s.chunk_count_accuracy_mean = np.mean(cca)
    s.chunk_count_accuracy_std = np.std(cca)

    cce = [pm.structural.chunk_count_error for pm in page_metrics_list]
    s.chunk_count_error_mean = np.mean(cce)
    s.chunk_count_error_std = np.std(cce)

    bf1 = [pm.structural.boundary_f1 for pm in page_metrics_list]
    s.boundary_f1_mean = np.mean(bf1)
    s.boundary_f1_std = np.std(bf1)
    s.boundary_precision_mean = np.mean([pm.structural.boundary_precision for pm in page_metrics_list])
    s.boundary_recall_mean = np.mean([pm.structural.boundary_recall for pm in page_metrics_list])
    s.boundary_avg_error_mean = np.mean([pm.structural.boundary_avg_error for pm in page_metrics_list])

    s.matched_content_precision_mean = np.mean([pm.content.matched_content_precision for pm in page_metrics_list])
    s.matched_content_recall_mean = np.mean([pm.content.matched_content_recall for pm in page_metrics_list])
    s.matched_content_f1_mean = np.mean([pm.content.matched_content_f1 for pm in page_metrics_list])
    s.matched_content_f1_std = np.std([pm.content.matched_content_f1 for pm in page_metrics_list])

    all_chunks = []
    for page in sorted(model_result.available_pages):
        if page in model_result.pages:
            all_chunks.extend(model_result.pages[page].chunks)
    bi_score, bi_detail = compute_boundary_integrity_document(all_chunks, sentences_gt)
    s.boundary_integrity_mean = bi_score
    s.boundary_integrity_std = 0.0
    s.boundary_integrity_detail = bi_detail

    weights = {
        "chunk_count_accuracy": 0.20,
        "boundary_f1": 0.30,
        "matched_content_f1": 0.30,
        "boundary_integrity": 0.20,
    }
    score = (
        weights["chunk_count_accuracy"] * s.chunk_count_accuracy_mean
        + weights["boundary_f1"] * s.boundary_f1_mean
        + weights["matched_content_f1"] * s.matched_content_f1_mean
        + weights["boundary_integrity"] * s.boundary_integrity_mean
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
                        "matched_content_precision": pm.content.matched_content_precision,
                        "matched_content_recall": pm.content.matched_content_recall,
                        "matched_content_f1": pm.content.matched_content_f1,
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
                    "chunk_count_error_mean": sm.chunk_count_error_mean,
                    "chunk_count_error_std": sm.chunk_count_error_std,
                    "boundary_f1_mean": sm.boundary_f1_mean,
                    "boundary_f1_std": sm.boundary_f1_std,
                    "boundary_precision_mean": sm.boundary_precision_mean,
                    "boundary_recall_mean": sm.boundary_recall_mean,
                    "boundary_avg_error_mean": sm.boundary_avg_error_mean,
                },
                "content": {
                    "matched_content_precision_mean": sm.matched_content_precision_mean,
                    "matched_content_recall_mean": sm.matched_content_recall_mean,
                    "matched_content_f1_mean": sm.matched_content_f1_mean,
                    "matched_content_f1_std": sm.matched_content_f1_std,
                    "boundary_integrity_mean": sm.boundary_integrity_mean,
                    "boundary_integrity_std": sm.boundary_integrity_std,
                    "boundary_integrity_detail": sm.boundary_integrity_detail,
                },
            }

        return {
            "model_names": self.model_names,
            "pages": self.pages,
            "page_results": page_results_serializable,
            "summaries": summaries_serializable,
        }


def run_evaluation(
    data: ExperimentData,
    sentences_gt: Optional[List[str]] = None,
) -> EvaluationResults:
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
        model_result = data.model_results[model_name]
        summaries[model_name] = _compute_summary(model_name, page_results[model_name], model_result, sentences_gt)

    return EvaluationResults(
        model_names=data.model_names,
        pages=data.pages,
        page_results=page_results,
        summaries=summaries,
    )