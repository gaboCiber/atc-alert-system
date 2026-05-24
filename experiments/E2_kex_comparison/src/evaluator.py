from dataclasses import dataclass, field
from typing import List, Dict, Optional
from tqdm import tqdm

from config import E2Config, JudgeConfig, MetricConfig
from loader import ExperimentData, KexPageResult
from matcher import match_all_types, MatchingOutput, KEX_TYPES
from metrics import PageMetrics, compute_page_metrics, PageTypeMetrics
from llm_judge import LLMJudge, Judgment


@dataclass
class SemanticScores:
    kex_type: str
    scores: List[float] = field(default_factory=list)
    explanations: List[str] = field(default_factory=list)
    mean_score: float = 0.0

    def __post_init__(self):
        if self.scores:
            self.mean_score = sum(self.scores) / len(self.scores)


@dataclass
class ModelPageMetrics:
    page: int
    model: str
    structural_metrics: PageMetrics
    semantic_scores: Dict[str, SemanticScores] = field(default_factory=dict)
    overall_semantic: float = 0.0
    overall_score: float = 0.0


@dataclass
class ModelSummaryMetrics:
    model: str
    page_metrics: List[ModelPageMetrics] = field(default_factory=list)

    structural_precision: float = 0.0
    structural_recall: float = 0.0
    structural_f1: float = 0.0
    content_avg: float = 0.0
    cross_ref_avg: float = 0.0
    semantic_avg: float = 0.0

    type_structural_f1: Dict[str, float] = field(default_factory=dict)
    type_content: Dict[str, float] = field(default_factory=dict)
    type_cross_ref: Dict[str, float] = field(default_factory=dict)
    type_semantic: Dict[str, float] = field(default_factory=dict)

    overall_score: float = 0.0


def _compute_summary(
    model_name: str,
    page_metrics_list: List[ModelPageMetrics],
    metric_cfg: MetricConfig,
) -> ModelSummaryMetrics:
    s = ModelSummaryMetrics(model=model_name, page_metrics=page_metrics_list)

    s.structural_precision = sum(pm.structural_metrics.overall_precision for pm in page_metrics_list) / max(len(page_metrics_list), 1)
    s.structural_recall = sum(pm.structural_metrics.overall_recall for pm in page_metrics_list) / max(len(page_metrics_list), 1)
    s.structural_f1 = sum(pm.structural_metrics.overall_f1 for pm in page_metrics_list) / max(len(page_metrics_list), 1)
    s.content_avg = sum(pm.structural_metrics.overall_content for pm in page_metrics_list) / max(len(page_metrics_list), 1)
    s.cross_ref_avg = sum(pm.structural_metrics.overall_cross_ref for pm in page_metrics_list) / max(len(page_metrics_list), 1)
    s.semantic_avg = sum(pm.overall_semantic for pm in page_metrics_list) / max(len(page_metrics_list), 1)

    for kex_type in KEX_TYPES:
        f1s = []
        cs = []
        crs = []
        ss = []
        for pm in page_metrics_list:
            tm = pm.structural_metrics.type_metrics.get(kex_type)
            if tm:
                f1s.append(tm.structural.f1)
                cs.append(tm.content.avg_field_match)
                crs.append(tm.cross_ref.validity_ratio)
            sem = pm.semantic_scores.get(kex_type)
            if sem:
                ss.append(sem.mean_score)

        s.type_structural_f1[kex_type] = sum(f1s) / len(f1s) if f1s else 0.0
        s.type_content[kex_type] = sum(cs) / len(cs) if cs else 0.0
        s.type_cross_ref[kex_type] = sum(crs) / len(crs) if crs else 0.0
        s.type_semantic[kex_type] = sum(ss) / len(ss) if ss else 0.0

    s.overall_score = (
        metric_cfg.structural_weight * s.structural_f1
        + metric_cfg.content_weight * s.content_avg
        + metric_cfg.cross_ref_weight * s.cross_ref_avg
        + metric_cfg.semantic_weight * s.semantic_avg
    )

    return s


@dataclass
class EvaluationResults:
    model_names: List[str]
    pages: List[int]
    page_results: Dict[str, List[ModelPageMetrics]]
    summaries: Dict[str, ModelSummaryMetrics]

    def to_dict(self) -> dict:
        page_results_serializable = {}
        for model, pms in self.page_results.items():
            page_results_serializable[model] = [
                {
                    "page": pm.page,
                    "model": pm.model,
                    "overall_score": pm.overall_score,
                    "overall_semantic": pm.overall_semantic,
                    "structural": {
                        "overall_precision": pm.structural_metrics.overall_precision,
                        "overall_recall": pm.structural_metrics.overall_recall,
                        "overall_f1": pm.structural_metrics.overall_f1,
                        "overall_content": pm.structural_metrics.overall_content,
                        "overall_cross_ref": pm.structural_metrics.overall_cross_ref,
                        "total_gt_items": pm.structural_metrics.total_gt_items,
                        "total_model_items": pm.structural_metrics.total_model_items,
                    },
                    "by_type": {
                        kex_type: {
                            "structural": {
                                "gt_count": tm.structural.gt_count,
                                "model_count": tm.structural.model_count,
                                "precision": tm.structural.precision,
                                "recall": tm.structural.recall,
                                "f1": tm.structural.f1,
                            },
                            "content": {
                                "avg_field_match": tm.content.avg_field_match,
                            },
                            "cross_ref": {
                                "total_refs": tm.cross_ref.total_refs,
                                "valid_refs": tm.cross_ref.valid_refs,
                                "validity_ratio": tm.cross_ref.validity_ratio,
                                "broken_refs": tm.cross_ref.broken_refs,
                            },
                            "semantic": {
                                "mean_score": pm.semantic_scores.get(kex_type, SemanticScores(kex_type)).mean_score,
                                "scores": pm.semantic_scores.get(kex_type, SemanticScores(kex_type)).scores,
                            },
                        }
                        for kex_type, tm in pm.structural_metrics.type_metrics.items()
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
                    "precision": sm.structural_precision,
                    "recall": sm.structural_recall,
                    "f1": sm.structural_f1,
                },
                "content_avg": sm.content_avg,
                "cross_ref_avg": sm.cross_ref_avg,
                "semantic_avg": sm.semantic_avg,
                "by_type": {
                    kex_type: {
                        "structural_f1": sm.type_structural_f1.get(kex_type, 0.0),
                        "content": sm.type_content.get(kex_type, 0.0),
                        "cross_ref": sm.type_cross_ref.get(kex_type, 0.0),
                        "semantic": sm.type_semantic.get(kex_type, 0.0),
                    }
                    for kex_type in KEX_TYPES
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
    judge: LLMJudge,
    metric_cfg: MetricConfig,
) -> EvaluationResults:
    page_results: Dict[str, List[ModelPageMetrics]] = {model: [] for model in data.model_names}

    for page in tqdm(data.pages, desc="Evaluating pages"):
        gt = data.ground_truth[page]

        for model_name in data.model_names:
            model_result = data.model_results[model_name]
            if page not in model_result.pages:
                continue
            model_page = model_result.pages[page]

            matchings = match_all_types(gt, model_page)
            structural = compute_page_metrics(page, gt, model_page, matchings)

            semantic_scores: Dict[str, SemanticScores] = {}
            for kex_type in KEX_TYPES:
                matching = matchings.get(kex_type)
                if not matching:
                    continue

                scores = []
                explanations = []
                for match in tqdm(matching.matches, desc=f"  Judging {kex_type} matches", leave=False):
                    judgment = judge.judge(kex_type, match.gt_item, match.model_item)
                    if judgment:
                        scores.append(judgment.similarity_score)
                        explanations.append(judgment.explanation)

                semantic_scores[kex_type] = SemanticScores(
                    kex_type=kex_type,
                    scores=scores,
                    explanations=explanations,
                )

            all_sem = [ss.mean_score for ss in semantic_scores.values() if ss.scores]
            overall_semantic = sum(all_sem) / len(all_sem) if all_sem else 0.0

            overall_score = (
                metric_cfg.structural_weight * structural.overall_f1
                + metric_cfg.content_weight * structural.overall_content
                + metric_cfg.cross_ref_weight * structural.overall_cross_ref
                + metric_cfg.semantic_weight * overall_semantic
            )

            mpm = ModelPageMetrics(
                page=page,
                model=model_name,
                structural_metrics=structural,
                semantic_scores=semantic_scores,
                overall_semantic=overall_semantic,
                overall_score=overall_score,
            )
            page_results[model_name].append(mpm)

    summaries: Dict[str, ModelSummaryMetrics] = {}
    for model_name in data.model_names:
        summaries[model_name] = _compute_summary(model_name, page_results[model_name], metric_cfg)

    return EvaluationResults(
        model_names=data.model_names,
        pages=data.pages,
        page_results=page_results,
        summaries=summaries,
    )
