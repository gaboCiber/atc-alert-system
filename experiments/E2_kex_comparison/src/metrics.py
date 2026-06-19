from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .matcher import MatchingOutput, KEX_TYPES
from .loader import KexPageResult


@dataclass
class StructuralMetrics:
    kex_type: str
    gt_count: int
    model_count: int
    count_error: int
    count_accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass
class ContentMetrics:
    kex_type: str
    exact_field_match_ratio: float
    enum_match_ratio: float
    avg_field_match: float


@dataclass
class CrossRefMetrics:
    kex_type: str
    total_refs: int
    valid_refs: int
    invalid_refs: int
    validity_ratio: float
    broken_refs: List[str] = field(default_factory=list)


@dataclass
class PageTypeMetrics:
    kex_type: str
    structural: StructuralMetrics
    content: ContentMetrics
    cross_ref: CrossRefMetrics


@dataclass
class PageMetrics:
    page: int
    type_metrics: Dict[str, PageTypeMetrics] = field(default_factory=dict)

    total_gt_items: int = 0
    total_model_items: int = 0
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    overall_f1: float = 0.0
    overall_content: float = 0.0
    overall_cross_ref: float = 0.0

    total_errors: int = 0
    extraction_failures: int = 0
    invalid_cross_refs: int = 0
    error_rate: float = 0.0
    chunks_with_errors: int = 0


ENUM_FIELDS = {
    "entities": [],
    "relationships": ["relation_type"],
    "events": ["phase"],
    "rules": ["rule_type", "modality", "deontic_strength", "severity"],
    "procedures": [],
}

CONTENT_FIELDS = {
    "entities": ["text", "label", "subtype", "context", "aliases"],
    "relationships": ["subject_text", "predicate", "object_text", "relation_type", "context"],
    "events": ["event_type", "trigger_text", "actors", "targets", "phase", "parameters", "temporal_context"],
    "rules": [
        "rule_type", "modality", "deontic_strength", "severity", "safety_critical", "explainability",
        "linked_entities", "linked_relationships", "preconditions",
    ],
    "procedures": ["name", "purpose", "context", "mandatory_order", "preconditions", "exceptions", "linked_rules"],
}


def _compute_content_metrics(gt_item: dict, model_item: dict, kex_type: str) -> ContentMetrics:
    fields = CONTENT_FIELDS.get(kex_type, [])
    enum_fields = ENUM_FIELDS.get(kex_type, [])

    if not fields:
        return ContentMetrics(kex_type=kex_type, exact_field_match_ratio=0.0, enum_match_ratio=0.0, avg_field_match=0.0)

    exact_matches = 0
    enum_matches = 0
    enum_total = 0
    total_fields = len(fields)

    for f in fields:
        gt_val = gt_item.get(f)
        model_val = model_item.get(f)

        if gt_val == model_val:
            exact_matches += 1

        if f in enum_fields:
            enum_total += 1
            if gt_val == model_val:
                enum_matches += 1

    exact_ratio = exact_matches / total_fields if total_fields > 0 else 0.0
    enum_ratio = enum_matches / enum_total if enum_total > 0 else 0.0
    avg = (exact_ratio + enum_ratio) / 2 if (exact_ratio + enum_ratio) > 0 else 0.0

    return ContentMetrics(
        kex_type=kex_type,
        exact_field_match_ratio=exact_ratio,
        enum_match_ratio=enum_ratio,
        avg_field_match=avg,
    )


def _validate_cross_refs(items: List[dict], kex_type: str, page: KexPageResult) -> CrossRefMetrics:
    all_entity_ids = {e.get("id") for e in page.entities if e.get("id")}
    all_rel_ids = {r.get("id") for r in page.relationships if r.get("id")}
    all_event_ids = {e.get("id") for e in page.events if e.get("id")}
    all_rule_ids = {r.get("id") for r in page.rules if r.get("id")}

    total_refs = 0
    valid_refs = 0
    broken: List[str] = []

    if kex_type == "relationships":
        for item in items:
            sid = item.get("subject_id")
            oid = item.get("object_id")
            if sid:
                total_refs += 1
                if sid in all_entity_ids:
                    valid_refs += 1
                else:
                    broken.append(f"{item.get('id', '?')}.subject_id={sid}")
            if oid:
                total_refs += 1
                if oid in all_entity_ids:
                    valid_refs += 1
                else:
                    broken.append(f"{item.get('id', '?')}.object_id={oid}")

    elif kex_type == "events":
        for item in items:
            for aid in item.get("actors", []):
                total_refs += 1
                if aid in all_entity_ids:
                    valid_refs += 1
                else:
                    broken.append(f"{item.get('id', '?')}.actor={aid}")
            for tid in item.get("targets", []):
                total_refs += 1
                if tid in all_entity_ids:
                    valid_refs += 1
                else:
                    broken.append(f"{item.get('id', '?')}.target={tid}")

    elif kex_type == "rules":
        for item in items:
            for eid in item.get("linked_entities", []):
                total_refs += 1
                if eid in all_entity_ids:
                    valid_refs += 1
                else:
                    broken.append(f"{item.get('id', '?')}.linked_entity={eid}")
            for rid in item.get("linked_relationships", []):
                total_refs += 1
                if rid in all_rel_ids:
                    valid_refs += 1
                else:
                    broken.append(f"{item.get('id', '?')}.linked_rel={rid}")

    elif kex_type == "procedures":
        for item in items:
            for step in item.get("steps", []):
                for eid in step.get("required_entities", []):
                    total_refs += 1
                    if eid in all_entity_ids:
                        valid_refs += 1
                    else:
                        broken.append(f"{item.get('id', '?')}.step.entity={eid}")
                for eid in step.get("required_events", []):
                    total_refs += 1
                    if eid in all_event_ids:
                        valid_refs += 1
                    else:
                        broken.append(f"{item.get('id', '?')}.step.event={eid}")
            for rid in item.get("linked_rules", []):
                total_refs += 1
                if rid in all_rule_ids:
                    valid_refs += 1
                else:
                    broken.append(f"{item.get('id', '?')}.linked_rule={rid}")

    validity = valid_refs / total_refs if total_refs > 0 else 1.0

    return CrossRefMetrics(
        kex_type=kex_type,
        total_refs=total_refs,
        valid_refs=valid_refs,
        invalid_refs=total_refs - valid_refs,
        validity_ratio=validity,
        broken_refs=broken,
    )


def compute_page_metrics(
    page: int,
    gt_page: KexPageResult,
    model_page: KexPageResult,
    matchings: Dict[str, MatchingOutput],
) -> PageMetrics:
    pm = PageMetrics(page=page)
    type_metrics: Dict[str, PageTypeMetrics] = {}

    total_gt = 0
    total_model = 0
    all_prec = []
    all_rec = []
    all_f1 = []
    all_content = []
    all_cross_ref = []

    for kex_type in KEX_TYPES:
        gt_items = gt_page.get_by_type(kex_type)
        model_items = model_page.get_by_type(kex_type)
        matching = matchings.get(kex_type)

        if matching is None:
            continue

        structural = StructuralMetrics(
            kex_type=kex_type,
            gt_count=len(gt_items),
            model_count=len(model_items),
            count_error=abs(len(gt_items) - len(model_items)),
            count_accuracy=1.0 - abs(len(gt_items) - len(model_items)) / max(len(gt_items), 1),
            precision=matching.precision,
            recall=matching.recall,
            f1=matching.f1,
        )

        content_scores = []
        for match in matching.matches:
            cm = _compute_content_metrics(match.gt_item, match.model_item, kex_type)
            content_scores.append(cm.avg_field_match)
        avg_content = sum(content_scores) / len(content_scores) if content_scores else 0.0

        content = ContentMetrics(
            kex_type=kex_type,
            exact_field_match_ratio=avg_content,
            enum_match_ratio=avg_content,
            avg_field_match=avg_content,
        )

        cross_ref = _validate_cross_refs(model_items, kex_type, model_page)

        tm = PageTypeMetrics(
            kex_type=kex_type,
            structural=structural,
            content=content,
            cross_ref=cross_ref,
        )
        type_metrics[kex_type] = tm

        total_gt += len(gt_items)
        total_model += len(model_items)
        all_prec.append(matching.precision)
        all_rec.append(matching.recall)
        all_f1.append(matching.f1)
        all_content.append(avg_content)
        all_cross_ref.append(cross_ref.validity_ratio)

    pm.type_metrics = type_metrics
    pm.total_gt_items = total_gt
    pm.total_model_items = total_model
    pm.overall_precision = sum(all_prec) / len(all_prec) if all_prec else 0.0
    pm.overall_recall = sum(all_rec) / len(all_rec) if all_rec else 0.0
    pm.overall_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
    pm.overall_content = sum(all_content) / len(all_content) if all_content else 0.0
    pm.overall_cross_ref = sum(all_cross_ref) / len(all_cross_ref) if all_cross_ref else 0.0

    # Compute error metrics from model_page.errors
    if model_page.errors:
        chunks_with_err = set()
        for err in model_page.errors:
            err_type = err.get("type", "")
            if err_type == "extraction_failed":
                pm.extraction_failures += 1
            elif err_type == "invalid_cross_reference":
                pm.invalid_cross_refs += 1
            ci = err.get("chunk_index")
            if ci is not None:
                chunks_with_err.add(ci)
        pm.total_errors = len(model_page.errors)
        pm.chunks_with_errors = len(chunks_with_err)
        pm.error_rate = pm.total_errors / max(pm.total_model_items, 1)

    return pm
