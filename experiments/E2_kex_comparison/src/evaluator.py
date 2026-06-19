from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
from tqdm import tqdm

from .config import E2Config, JudgeConfig, MetricConfig, DedupConfig, get_config_hash
from .loader import ExperimentData, KexPageResult
from .matcher import match_all_types, MatchingOutput, KEX_TYPES
from .metrics import PageMetrics, compute_page_metrics, PageTypeMetrics
from .llm_judge import LLMJudge, Judgment
from .dedup import DedupReport, analyze_model
from .checkpoint import save_checkpoint, load_checkpoint, HolisticCheckpointer


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
    dedup_score: float = 0.0
    error_score: float = 1.0
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
    dedup_avg: float = 0.0
    error_avg: float = 0.0

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
    s.dedup_avg = sum(pm.dedup_score for pm in page_metrics_list) / max(len(page_metrics_list), 1)
    s.error_avg = sum(pm.error_score for pm in page_metrics_list) / max(len(page_metrics_list), 1)

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
        + metric_cfg.error_weight * s.error_avg
    )

    return s


@dataclass
class EvaluationResults:
    model_names: List[str]
    pages: List[int]
    page_results: Dict[str, List[ModelPageMetrics]]
    summaries: Dict[str, ModelSummaryMetrics]
    dedup_reports: Dict[str, DedupReport] = field(default_factory=dict)

    def to_dict(self) -> dict:
        page_results_serializable = {}
        for model, pms in self.page_results.items():
            page_results_serializable[model] = [
                {
                    "page": pm.page,
                    "model": pm.model,
                    "overall_score": pm.overall_score,
                    "overall_semantic": pm.overall_semantic,
                    "dedup_score": pm.dedup_score,
                    "error_score": pm.error_score,
                    "structural": {
                        "overall_precision": pm.structural_metrics.overall_precision,
                        "overall_recall": pm.structural_metrics.overall_recall,
                        "overall_f1": pm.structural_metrics.overall_f1,
                        "overall_content": pm.structural_metrics.overall_content,
                        "overall_cross_ref": pm.structural_metrics.overall_cross_ref,
                        "total_gt_items": pm.structural_metrics.total_gt_items,
                        "total_model_items": pm.structural_metrics.total_model_items,
                        "total_errors": pm.structural_metrics.total_errors,
                        "extraction_failures": pm.structural_metrics.extraction_failures,
                        "invalid_cross_refs": pm.structural_metrics.invalid_cross_refs,
                        "error_rate": pm.structural_metrics.error_rate,
                        "chunks_with_errors": pm.structural_metrics.chunks_with_errors,
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
                "dedup_avg": sm.dedup_avg,
                "error_avg": sm.error_avg,
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
            "dedup_reports": {
                model: report.to_dict() for model, report in self.dedup_reports.items()
            },
        }


def run_evaluation(
    data: ExperimentData,
    judge: LLMJudge,
    metric_cfg: MetricConfig,
    dedup_cfg: Optional[DedupConfig] = None,
    e2_cfg: Optional[E2Config] = None,
) -> EvaluationResults:
    # Pre-compute dedup reports if dedup is enabled
    dedup_reports: Dict[str, DedupReport] = {}
    if dedup_cfg and dedup_cfg.enabled and judge.client is not None:
        print("\nRunning dedup analysis...")
        for model_name in tqdm(data.model_names, desc="Dedup models"):
            # Try to load dedup checkpoint first
            config_hash = get_config_hash(judge.config, metric_cfg, dedup_cfg)
            dedup_checkpoint_data = load_checkpoint(
                e2_cfg.results_dir,
                "dedup",
                model_name,
                config_hash
            )
            
            if dedup_checkpoint_data is not None:
                # Reconstruct DedupReport from checkpointed data
                report = DedupReport.from_dict(dedup_checkpoint_data)
                dedup_reports[model_name] = report
                continue
            
            # Compute dedup analysis for this model (with intra-model checkpointing)
            model_result = data.model_results[model_name]
            report = analyze_model(
                model_result=model_result,
                judge=judge,
                batch_size=dedup_cfg.batch_size,
                threshold=dedup_cfg.threshold,
                results_dir=e2_cfg.results_dir,
                config_hash=config_hash,
            )
            dedup_reports[model_name] = report
            
            # Save outer checkpoint for this model's dedup report (marks full completion)
            save_checkpoint(
                e2_cfg.results_dir,
                "dedup",
                model_name,
                report.to_dict(),
                config_hash
            )
    
    page_results: Dict[str, List[ModelPageMetrics]] = {model: [] for model in data.model_names}
    
    for page in tqdm(data.pages, desc="Evaluating pages"):
        gt = data.ground_truth[page]
        
        for model_name in data.model_names:
            model_result = data.model_results[model_name]
            if page not in model_result.pages:
                print(f"DEBUG: Page {page} not in model {model_name} pages: {list(model_result.pages.keys())}")
                continue
            model_page = model_result.pages[page]
            print(f"DEBUG: Evaluating page {page} for model {model_name}")
            
            matchings = match_all_types(gt, model_page)
            structural = compute_page_metrics(page, gt, model_page, matchings)
            
            semantic_scores: Dict[str, SemanticScores] = {}
            
            # Get config hash for checkpoint validation
            config_hash = get_config_hash(judge.config, metric_cfg, dedup_cfg)
            
            # Try to load checkpoint for this page
            page_checkpoint_data = load_checkpoint(
                e2_cfg.results_dir, 
                f"holistic_page_{model_name}", 
                str(page), 
                config_hash
            )
            print(f"DEBUG: For {model_name} page {page}, checkpoint data: {'found' if page_checkpoint_data is not None else 'not found'}")
            
            if page_checkpoint_data is not None:
                # Use checkpointed results
                semantic_scores = {}
                for kex_type, score_data in page_checkpoint_data["semantic_scores"].items():
                    semantic_scores[kex_type] = SemanticScores(
                        kex_type=kex_type,
                        scores=score_data["scores"],
                        explanations=score_data["explanations"],
                    )
                overall_semantic = page_checkpoint_data["overall_semantic"]
            else:
                max_chunks = max(gt.chunk_count(), model_page.chunk_count())
                tracker = HolisticCheckpointer(
                    e2_cfg.results_dir, model_name, page, config_hash
                )

                per_type_chunk_data: Dict[int, Dict[str, Dict[str, Any]]] = {}

                for chunk_idx in range(max_chunks):
                    if tracker.is_chunk_completed(chunk_idx):
                        per_type_chunk_data[chunk_idx] = tracker.get_chunk_data(chunk_idx)
                        continue

                    gt_chunk_items_by_type = {}
                    model_chunk_items_by_type = {}
                    for kex_type in KEX_TYPES:
                        gt_chunk_items_by_type[kex_type] = gt.get_chunk_by_type(chunk_idx, kex_type)
                        model_chunk_items_by_type[kex_type] = model_page.get_chunk_by_type(chunk_idx, kex_type)

                    chunk_type_data: Dict[str, Dict[str, Any]] = {}
                    for kex_type in KEX_TYPES:
                        gt_items = gt_chunk_items_by_type[kex_type]
                        model_items = model_chunk_items_by_type[kex_type]
                        judgment = judge.holistic_judge(kex_type, gt_items, model_items)
                        if judgment:
                            chunk_type_data[kex_type] = {
                                "score": judgment.similarity_score,
                                "explanation": judgment.explanation,
                            }
                        else:
                            chunk_type_data[kex_type] = {
                                "score": None,
                                "explanation": None,
                            }

                    tracker.mark_chunk_completed(chunk_idx, chunk_type_data)
                    per_type_chunk_data[chunk_idx] = chunk_type_data

                # Aggregate per-type scores from all chunks (single pass)
                type_scores_agg: Dict[str, List[float]] = {kt: [] for kt in KEX_TYPES}
                type_expl_agg: Dict[str, List[str]] = {kt: [] for kt in KEX_TYPES}
                chunk_semantic_scores = []

                for chunk_idx in range(max_chunks):
                    chunk_type_data = per_type_chunk_data[chunk_idx]
                    chunk_type_scores = []
                    for kex_type in KEX_TYPES:
                        td = chunk_type_data.get(kex_type, {})
                        score = td.get("score")
                        expl = td.get("explanation")
                        if score is not None and expl is not None:
                            type_scores_agg[kex_type].append(score)
                            type_expl_agg[kex_type].append(expl)
                            chunk_type_scores.append(score)
                    if chunk_type_scores:
                        chunk_semantic_scores.append(
                            sum(chunk_type_scores) / len(chunk_type_scores)
                        )

                overall_semantic = (
                    sum(chunk_semantic_scores) / len(chunk_semantic_scores)
                    if chunk_semantic_scores else 0.0
                )

                for kex_type in KEX_TYPES:
                    semantic_scores[kex_type] = SemanticScores(
                        kex_type=kex_type,
                        scores=type_scores_agg[kex_type],
                        explanations=type_expl_agg[kex_type],
                    )

                # Save page-level checkpoint (only when judge is enabled)
                if judge.config.enabled:
                    checkpoint_data = {
                        "semantic_scores": {
                            k: {"scores": ss.scores, "explanations": ss.explanations, "mean_score": ss.mean_score}
                            for k, ss in semantic_scores.items()
                        },
                        "overall_semantic": overall_semantic,
                    }
                    save_checkpoint(
                        e2_cfg.results_dir,
                        f"holistic_page_{model_name}",
                        str(page),
                        checkpoint_data,
                        config_hash
                    )
                if judge.config.enabled:
                    tracker.cleanup()
            
            # Calculate dedup score for this model-page if dedup is enabled
            dedup_score = 0.0
            if dedup_cfg and dedup_cfg.enabled and model_name in dedup_reports:
                report = dedup_reports[model_name]
                # Normalize dedup rate to a score (lower dedup rate = higher score)
                # overall_duplication_rate is percentage of items that are duplicates (0-1)
                # We want: 0% dup -> 1.0 score, 100% dup -> 0.0 score
                dedup_score = 1.0 - min(report.overall_duplication_rate, 1.0)

            # Compute error_score: 1.0 = no errors, 0.0 = all items are errors
            error_score = 1.0 - min(structural.total_errors / max(structural.total_model_items, 1), 1.0)

            overall_score = (
                metric_cfg.structural_weight * structural.overall_f1
                + metric_cfg.content_weight * structural.overall_content
                + metric_cfg.cross_ref_weight * structural.overall_cross_ref
                + metric_cfg.semantic_weight * overall_semantic
                + metric_cfg.error_weight * error_score
            )

            mpm = ModelPageMetrics(
                page=page,
                model=model_name,
                structural_metrics=structural,
                semantic_scores=semantic_scores,
                overall_semantic=overall_semantic,
                dedup_score=dedup_score,
                error_score=error_score,
                overall_score=overall_score,
            )
            page_results[model_name].append(mpm)
    
    # Compute summary metrics for each model
    summaries: Dict[str, ModelSummaryMetrics] = {}
    for model_name in data.model_names:
        summaries[model_name] = _compute_summary(model_name, page_results[model_name], metric_cfg)
    
    return EvaluationResults(
        model_names=data.model_names,
        pages=data.pages,
        page_results=page_results,
        summaries=summaries,
        dedup_reports=dedup_reports,
    )