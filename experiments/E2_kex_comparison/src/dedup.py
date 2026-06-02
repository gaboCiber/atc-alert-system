from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from tqdm import tqdm

from loader import KexPageResult, ModelResult
from llm_judge import LLMJudge, CandidateJudgment
from dedup_prompts import FIELD_EXTRACTORS

KEX_TYPES = ["entities", "relationships", "events", "rules", "procedures"]


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def get_clusters(self) -> List[List[int]]:
        groups: Dict[int, List[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            groups.setdefault(root, []).append(i)
        return [g for g in groups.values() if len(g) > 1]


@dataclass
class DedupItem:
    global_id: str
    page: int
    item: Dict[str, Any]

    @property
    def text_preview(self) -> str:
        return self.item.get("text", self.item.get("name", self.item.get("id", "")))


@dataclass
class DedupCluster:
    items: List[DedupItem]
    size: int
    avg_similarity: float

    def to_dict(self) -> dict:
        return {
            "size": self.size,
            "avg_similarity": round(self.avg_similarity, 4),
            "items": [
                {
                    "global_id": it.global_id,
                    "page": it.page,
                    **{k: it.item.get(k) for k in ["text", "label", "name", "subject_text", "predicate", "object_text", "trigger_text", "event_type", "rule_type", "modality", "purpose"] if k in it.item}
                }
                for it in self.items
            ]
        }


@dataclass
class DedupTypeMetrics:
    kex_type: str
    total_items: int
    clusters: List[DedupCluster]
    unique_items: int = 0
    duplicate_count: int = 0
    cluster_count: int = 0
    max_cluster_size: int = 0
    duplication_rate: float = 0.0

    def __post_init__(self):
        dups = sum(c.size for c in self.clusters)
        self.duplicate_count = dups
        self.unique_items = self.total_items - dups
        self.cluster_count = len(self.clusters)
        self.max_cluster_size = max((c.size for c in self.clusters), default=0)
        self.duplication_rate = dups / self.total_items if self.total_items > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "kex_type": self.kex_type,
            "total_items": self.total_items,
            "unique_items": self.unique_items,
            "duplicate_count": self.duplicate_count,
            "cluster_count": self.cluster_count,
            "max_cluster_size": self.max_cluster_size,
            "duplication_rate": round(self.duplication_rate, 4),
            "clusters": [c.to_dict() for c in self.clusters],
        }


@dataclass
class DedupReport:
    model_name: str
    by_type: Dict[str, DedupTypeMetrics]
    overall_duplication_rate: float = 0.0

    def __post_init__(self):
        total = sum(t.total_items for t in self.by_type.values())
        dups = sum(t.duplicate_count for t in self.by_type.values())
        self.overall_duplication_rate = dups / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "overall_duplication_rate": round(self.overall_duplication_rate, 4),
            "by_type": {k: v.to_dict() for k, v in self.by_type.items()},
        }


def _format_exact_match_key(item: dict, kex_type: str) -> str:
    if kex_type == "entities":
        return (item.get("text", "") or "").strip().lower()
    elif kex_type == "relationships":
        parts = [item.get("subject_text", ""), item.get("predicate", ""), item.get("object_text", "")]
        return "|".join(p.strip().lower() for p in parts)
    elif kex_type == "events":
        return (item.get("trigger_text", "") or "").strip().lower()
    elif kex_type == "rules":
        trigger = item.get("trigger", {})
        desc = trigger.get("description", "") if isinstance(trigger, dict) else str(trigger)
        return desc.strip().lower()
    elif kex_type == "procedures":
        return (item.get("name", "") or "").strip().lower()
    return ""


def collect_items_cross_page(model_result: ModelResult, kex_type: str) -> List[DedupItem]:
    items: List[DedupItem] = []
    for page in sorted(model_result.available_pages):
        page_result = model_result.pages[page]
        raw_items = page_result.get_by_type(kex_type)
        for raw in raw_items:
            item_id = raw.get("id", "unknown")
            items.append(DedupItem(
                global_id=f"pag_{page}/{item_id}",
                page=page,
                item=raw,
            ))
    return items


def _short_circuit_exact(items: List[DedupItem], kex_type: str) -> UnionFind:
    uf = UnionFind(len(items))
    keys: List[str] = []
    for di in items:
        keys.append(_format_exact_match_key(di.item, kex_type))

    seen: Dict[str, int] = {}
    for i, key in enumerate(keys):
        if key:
            if key in seen:
                uf.union(seen[key], i)
            else:
                seen[key] = i
    return uf


def build_clusters(
    items: List[DedupItem],
    kex_type: str,
    judge: LLMJudge,
    batch_size: int = 10,
    threshold: float = 0.80,
    exact_uf: Optional[UnionFind] = None,
) -> DedupTypeMetrics:
    n = len(items)

    if n <= 1:
        return DedupTypeMetrics(kex_type=kex_type, total_items=n, clusters=[])

    uf = UnionFind(n)
    if exact_uf:
        for i in range(n):
            uf.parent[i] = exact_uf.find(i)
            uf.rank[i] = exact_uf.rank[i]

    edge_scores: Dict[tuple, float] = {}

    if exact_uf:
        for i in range(n):
            for j in range(i + 1, n):
                if exact_uf.find(i) == exact_uf.find(j):
                    edge_scores[(i, j)] = 1.0

    for ref_idx in range(n - 1):
        ref_item = items[ref_idx]

        cand_start = ref_idx + 1
        if cand_start >= n:
            continue

        for batch_start in range(cand_start, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_candidates = items[batch_start:batch_end]

            already_same_cluster = True
            for ci in batch_candidates:
                if uf.find(ref_idx) != uf.find(items.index(ci)):
                    already_same_cluster = False
                    break
            if already_same_cluster:
                continue

            result = judge.batch_judge(
                kex_type,
                ref_item.item,
                [ci.item for ci in batch_candidates],
            )

            if result is None or not result.judgments:
                continue

            for j, jdg in enumerate(result.judgments):
                cand_idx = batch_start + j
                if cand_idx >= n:
                    continue
                score = jdg.similarity_score
                score_key = (ref_idx, cand_idx)
                edge_scores[score_key] = score
                if jdg.is_duplicate and score >= threshold:
                    uf.union(ref_idx, cand_idx)

    clusters = uf.get_clusters()
    dedup_clusters: List[DedupCluster] = []

    for cl in clusters:
        scores = []
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                key = (min(cl[i], cl[j]), max(cl[i], cl[j]))
                s = edge_scores.get(key)
                if s is not None:
                    scores.append(s)
        avg_sim = sum(scores) / len(scores) if scores else 0.0
        dedup_clusters.append(DedupCluster(
            items=[items[idx] for idx in cl],
            size=len(cl),
            avg_similarity=avg_sim,
        ))

    return DedupTypeMetrics(kex_type=kex_type, total_items=n, clusters=dedup_clusters)


def analyze_model(
    model_result: ModelResult,
    judge: LLMJudge,
    batch_size: int = 10,
    threshold: float = 0.80,
) -> DedupReport:
    by_type: Dict[str, DedupTypeMetrics] = {}

    for kex_type in tqdm(KEX_TYPES, desc=f"  Dedup {model_result.model_name}"):
        items = collect_items_cross_page(model_result, kex_type)

        if not items:
            by_type[kex_type] = DedupTypeMetrics(kex_type=kex_type, total_items=0, clusters=[])
            continue

        exact_uf = _short_circuit_exact(items, kex_type)

        metrics = build_clusters(
            items=items,
            kex_type=kex_type,
            judge=judge,
            batch_size=batch_size,
            threshold=threshold,
            exact_uf=exact_uf,
        )
        by_type[kex_type] = metrics

    return DedupReport(model_name=model_result.model_name, by_type=by_type)


def format_card(report: DedupReport) -> str:
    lines = []
    sep = "=" * 80
    lines.append(sep)
    lines.append(f"Reporte de Duplicados Semánticos: {report.model_name}")
    lines.append(sep)
    lines.append(f"Tasa global: {report.overall_duplication_rate:.2%}")
    lines.append("")

    for kex_type in KEX_TYPES:
        m = report.by_type.get(kex_type)
        if not m or m.total_items == 0:
            continue
        lines.append(f"{'─' * 80}")
        lines.append(f"{kex_type.upper()} (tasa: {m.duplication_rate:.2%}, {m.duplicate_count}/{m.total_items} duplicados)")
        lines.append(f"")
        for cluster in m.clusters:
            items_str = "\n".join(
                f"    {it.global_id:25s} {it.text_preview:40s}"
                for it in cluster.items
            )
            lines.append(f"  Cluster ({cluster.size} items, sim: {cluster.avg_similarity:.2f}):")
            lines.append(items_str)
            lines.append("")

    if all(m.total_items == 0 for m in report.by_type.values()):
        lines.append("(sin datos)")

    return "\n".join(lines)
