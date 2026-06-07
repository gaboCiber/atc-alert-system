from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from rapidfuzz import fuzz

KEX_TYPES = ["entities", "relationships", "events", "rules", "procedures"]

FIELD_MAP = {
    "entities": "text",
    "relationships": "full_key",
    "events": "trigger_text",
    "rules": "trigger_desc",
    "procedures": "name",
}


def _get_match_key(item: dict, kex_type: str) -> str:
    if kex_type == "entities":
        return item.get("text", "")
    elif kex_type == "relationships":
        subj = item.get("subject_text", "")
        pred = item.get("predicate", "")
        obj = item.get("object_text", "")
        return f"{subj} {pred} {obj}"
    elif kex_type == "events":
        return item.get("trigger_text", "")
    elif kex_type == "rules":
        trigger = item.get("trigger", {})
        if isinstance(trigger, dict):
            return trigger.get("description", "")
        return str(trigger)
    elif kex_type == "procedures":
        return item.get("name", "")
    return ""


@dataclass
class MatchResult:
    gt_index: int
    model_index: int
    score: float
    gt_item: dict
    model_item: dict


@dataclass
class MatchingOutput:
    matches: List[MatchResult]
    gt_unmatched: List[int]
    model_unmatched: List[int]
    precision: float
    recall: float
    f1: float


def compute_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.ratio(a.lower().strip(), b.lower().strip()) / 100.0


def match_items(
    gt_items: List[dict],
    model_items: List[dict],
    kex_type: str,
    threshold: float = 0.70,
) -> MatchingOutput:
    if not gt_items and not model_items:
        return MatchingOutput([], [], [], 0.0, 0.0, 0.0)

    if not gt_items:
        return MatchingOutput([], [], list(range(len(model_items))), 0.0, 0.0, 0.0)

    if not model_items:
        return MatchingOutput([], list(range(len(gt_items))), [], 0.0, 0.0, 0.0)

    gt_keys = [_get_match_key(item, kex_type) for item in gt_items]
    model_keys = [_get_match_key(item, kex_type) for item in model_items]

    n_gt = len(gt_items)
    n_model = len(model_items)

    score_matrix = [[0.0] * n_model for _ in range(n_gt)]
    for i in range(n_gt):
        for j in range(n_model):
            score_matrix[i][j] = compute_similarity(gt_keys[i], model_keys[j])

    matched_gt = set()
    matched_model = set()
    matches: List[MatchResult] = []

    pairs = []
    for i in range(n_gt):
        for j in range(n_model):
            pairs.append((i, j, score_matrix[i][j]))
    pairs.sort(key=lambda x: -x[2])

    for i, j, score in pairs:
        if score < threshold:
            break
        if i not in matched_gt and j not in matched_model:
            matches.append(MatchResult(
                gt_index=i,
                model_index=j,
                score=score,
                gt_item=gt_items[i],
                model_item=model_items[j],
            ))
            matched_gt.add(i)
            matched_model.add(j)

    gt_unmatched = [i for i in range(n_gt) if i not in matched_gt]
    model_unmatched = [j for j in range(n_model) if j not in matched_model]

    tp = len(matches)
    fp = len(model_unmatched)
    fn = len(gt_unmatched)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return MatchingOutput(
        matches=matches,
        gt_unmatched=gt_unmatched,
        model_unmatched=model_unmatched,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def match_all_types(
    gt_page: "KexPageResult",
    model_page: "KexPageResult",
    threshold: float = 0.70,
) -> Dict[str, MatchingOutput]:
    from .loader import KexPageResult

    results: Dict[str, MatchingOutput] = {}
    for kex_type in KEX_TYPES:
        gt_items = gt_page.get_by_type(kex_type)
        model_items = model_page.get_by_type(kex_type)
        results[kex_type] = match_items(gt_items, model_items, kex_type, threshold)

    return results
