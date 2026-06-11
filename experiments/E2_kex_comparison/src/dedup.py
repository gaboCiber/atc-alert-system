import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Set, Callable
from tqdm import tqdm

from .loader import KexPageResult, ModelResult
from .llm_judge import LLMJudge, CandidateJudgment
from .dedup_prompts import FIELD_EXTRACTORS

KEX_TYPES = ["entities", "relationships", "events", "rules", "procedures"]


class DedupInterruptedError(Exception):
    """Raised when dedup analysis is interrupted by an LLM call failure."""
    pass


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
                    "global_id": it["global_id"] if isinstance(it, dict) else it.global_id,
                    "page": it["page"] if isinstance(it, dict) else it.page,
                    **{k: it["item"].get(k) if isinstance(it, dict) else it.item.get(k) for k in ["text", "label", "name", "subject_text", "predicate", "object_text", "trigger_text", "event_type", "rule_type", "modality", "purpose"] if (k in it["item"] if isinstance(it, dict) else k in it.item)}
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

    @classmethod
    def from_dict(cls, data: dict) -> "DedupTypeMetrics":
        clusters = []
        for cluster_data in data.get("clusters", []):
            items = []
            for item_data in cluster_data.get("items", []):
                item_fields = {k: v for k, v in item_data.items()
                               if k not in ["global_id", "page"]}
                item = DedupItem(
                    global_id=item_data["global_id"],
                    page=item_data["page"],
                    item=item_fields,
                )
                items.append(item)
            cluster = DedupCluster(
                items=items,
                size=cluster_data.get("size", 0),
                avg_similarity=cluster_data.get("avg_similarity", 0.0),
            )
            clusters.append(cluster)
        return cls(
            kex_type=data.get("kex_type", ""),
            total_items=data.get("total_items", 0),
            clusters=clusters,
        )


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

    @classmethod
    def from_dict(cls, data: dict) -> "DedupReport":
        """Reconstruct a DedupReport from a dictionary (e.g., from a checkpoint)."""
        # Import inside method to avoid circular imports
        from .dedup import DedupTypeMetrics, DedupCluster, DedupItem
        
        by_type = {}
        for kex_type, type_data in data.get("by_type", {}).items():
            # Reconstruct clusters
            clusters = []
            for cluster_data in type_data.get("clusters", []):
                # Reconstruct DedupItem objects from the stored item data
                items = []
                for item_data in cluster_data.get("items", []):
                    # Extract the item fields (everything except global_id and page)
                    item_fields = {k: v for k, v in item_data.items() 
                                 if k not in ["global_id", "page"]}
                    # Create DedupItem
                    item = DedupItem(
                        global_id=item_data["global_id"],
                        page=item_data["page"],
                        item=item_fields
                    )
                    items.append(item)
                
                cluster = DedupCluster(
                    items=items,
                    size=cluster_data.get("size", 0),
                    avg_similarity=cluster_data.get("avg_similarity", 0.0)
                )
                clusters.append(cluster)
            
            # Reconstruct DedupTypeMetrics
            type_metrics = DedupTypeMetrics(
                kex_type=kex_type,
                total_items=type_data.get("total_items", 0),
                clusters=clusters
                # Note: unique_items, duplicate_count, etc. are computed in __post_init__
            )
            by_type[kex_type] = type_metrics
        
        # Create the DedupReport
        report = cls(
            model_name=data.get("model", ""),
            by_type=by_type
        )
        # The overall_duplication_rate will be computed in __post_init__
        return report


class DedupCheckpointer:
    """Manages intra-model checkpointing for dedup analysis.

    Saves progress after each completed KEX type and after each batch LLM
    call within build_clusters(), enabling resume after interruption.
    """
    def __init__(self, results_dir: Path, model_name: str, config_hash: str):
        self.path = results_dir / "checkpoints" / f"dedup_intra_{model_name}.json"
        self.model_name = model_name
        self.config_hash = config_hash
        self.completed_types: Set[str] = set()
        self.by_type_metrics: Dict[str, dict] = {}
        self.in_progress: Optional[dict] = None
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("config_hash") != self.config_hash:
                return
            self.completed_types = set(data.get("completed_types", []))
            self.by_type_metrics = data.get("by_type_metrics", {})
            self.in_progress = data.get("in_progress")
        except (json.JSONDecodeError, KeyError, IOError):
            pass

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {
            "config_hash": self.config_hash,
            "model_name": self.model_name,
            "completed_types": list(self.completed_types),
            "by_type_metrics": self.by_type_metrics,
        }
        if self.in_progress is not None:
            data["in_progress"] = self.in_progress
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_metrics(self, kex_type: str) -> Optional["DedupTypeMetrics"]:
        d = self.by_type_metrics.get(kex_type)
        if d is not None:
            return DedupTypeMetrics.from_dict(d)
        return None

    def get_build_resume_state(self, kex_type: str) -> Optional[dict]:
        if self.in_progress and self.in_progress.get("kex_type") == kex_type:
            return dict(self.in_progress.get("build_state", {}))
        return None

    def mark_type_completed(self, kex_type: str, metrics: "DedupTypeMetrics"):
        self.completed_types.add(kex_type)
        self.by_type_metrics[kex_type] = metrics.to_dict()
        self.in_progress = None
        self._save()

    def save_batch_progress(self, kex_type: str, build_state: dict):
        self.in_progress = {
            "kex_type": kex_type,
            "build_state": build_state,
        }
        self._save()

    def cleanup(self):
        if self.path.exists():
            self.path.unlink()


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


def _parse_edge_scores(data: Dict[str, float]) -> Dict[tuple, float]:
    result: Dict[tuple, float] = {}
    for k, v in data.items():
        stripped = k.strip("()")
        parts = stripped.split(",")
        if len(parts) == 2:
            result[(int(parts[0]), int(parts[1]))] = v
    return result


def _serialize_edge_scores(scores: Dict[tuple, float]) -> Dict[str, float]:
    return {f"({i},{j})": v for (i, j), v in scores.items()}


def build_clusters(
    items: List[DedupItem],
    kex_type: str,
    judge: LLMJudge,
    batch_size: int = 10,
    threshold: float = 0.80,
    exact_uf: Optional[UnionFind] = None,
    resume_state: Optional[dict] = None,
    on_batch_complete: Optional[Callable[[dict], None]] = None,
) -> DedupTypeMetrics:
    n = len(items)

    if n <= 1:
        return DedupTypeMetrics(kex_type=kex_type, total_items=n, clusters=[])

    uf = UnionFind(n)
    edge_scores: Dict[tuple, float] = {}
    completed: Set[tuple] = set()

    if resume_state is not None:
        if resume_state.get("items_count", n) == n:
            uf.parent = list(resume_state.get("uf_parent", list(range(n))))
            uf.rank = list(resume_state.get("uf_rank", [0] * n))
            edge_scores = _parse_edge_scores(resume_state.get("edge_scores", {}))
            completed = set(tuple(p) for p in resume_state.get("completed_batches", []))

    if not completed:
        if exact_uf:
            for i in range(n):
                uf.parent[i] = exact_uf.find(i)
                uf.rank[i] = exact_uf.rank[i]
            for i in range(n):
                for j in range(i + 1, n):
                    if exact_uf.find(i) == exact_uf.find(j):
                        edge_scores[(i, j)] = 1.0

    processed_roots: Set[int] = set()

    for ref_idx in range(n - 1):
        if uf.find(ref_idx) in processed_roots:
            continue
        processed_roots.add(uf.find(ref_idx))

        ref_item = items[ref_idx]

        cand_start = ref_idx + 1
        if cand_start >= n:
            continue

        for batch_start in range(cand_start, n, batch_size):
            batch_end = min(batch_start + batch_size, n)

            if (ref_idx, batch_start) in completed:
                continue

            batch_candidates = items[batch_start:batch_end]

            already_same_cluster = True
            for ci in batch_candidates:
                if uf.find(ref_idx) != uf.find(items.index(ci)):
                    already_same_cluster = False
                    break
            if already_same_cluster:
                completed.add((ref_idx, batch_start))
                continue

            try:
                result = judge.batch_judge(
                    kex_type,
                    ref_item.item,
                    [ci.item for ci in batch_candidates],
                )
            except Exception as e:
                # Save checkpoint before propagating
                if on_batch_complete:
                    on_batch_complete({
                        "uf_parent": list(uf.parent),
                        "uf_rank": list(uf.rank),
                        "edge_scores": _serialize_edge_scores(edge_scores),
                        "completed_batches": [list(p) for p in completed],
                        "items_count": n,
                    })
                raise DedupInterruptedError(
                    f"LLM call failed at {kex_type} ref={ref_idx} batch={batch_start}: {e}"
                ) from e

            if result is None or not result.judgments:
                completed.add((ref_idx, batch_start))
                if on_batch_complete:
                    on_batch_complete({
                        "uf_parent": list(uf.parent),
                        "uf_rank": list(uf.rank),
                        "edge_scores": _serialize_edge_scores(edge_scores),
                        "completed_batches": [list(p) for p in completed],
                        "items_count": n,
                    })
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

            completed.add((ref_idx, batch_start))
            if on_batch_complete:
                on_batch_complete({
                    "uf_parent": list(uf.parent),
                    "uf_rank": list(uf.rank),
                    "edge_scores": _serialize_edge_scores(edge_scores),
                    "completed_batches": [list(p) for p in completed],
                    "items_count": n,
                })

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
    results_dir: Optional[Path] = None,
    config_hash: Optional[str] = None,
    skip_types: Optional[set] = None,
) -> DedupReport:
    model_name = model_result.model_name
    by_type: Dict[str, DedupTypeMetrics] = {}
    skip_types = skip_types or set()

    ckptr = None
    if results_dir and config_hash:
        ckptr = DedupCheckpointer(results_dir, model_name, config_hash)
        for kt in list(ckptr.completed_types):
            if kt in skip_types:
                ckptr.completed_types.discard(kt)
                continue
            m = ckptr.get_metrics(kt)
            if m is not None:
                by_type[kt] = m
        if ckptr.in_progress and ckptr.in_progress.get("kex_type") in skip_types:
            ckptr.in_progress = None

    completed = set(by_type.keys())

    for kex_type in skip_types:
        by_type[kex_type] = DedupTypeMetrics(kex_type=kex_type, total_items=0, clusters=[])

    pbar = tqdm(total=len(KEX_TYPES), desc=f"  Dedup {model_name}", initial=len(completed))
    for kex_type in KEX_TYPES:
        if kex_type in completed:
            pbar.update(1)
            continue
        if kex_type in skip_types:
            pbar.update(1)
            continue

        items = collect_items_cross_page(model_result, kex_type)

        if not items:
            by_type[kex_type] = DedupTypeMetrics(kex_type=kex_type, total_items=0, clusters=[])
        else:
            exact_uf = _short_circuit_exact(items, kex_type)
            resume_state = ckptr.get_build_resume_state(kex_type) if ckptr else None

            def _on_batch(state):
                if ckptr is not None:
                    ckptr.save_batch_progress(kex_type, state)

            metrics = build_clusters(
                items=items,
                kex_type=kex_type,
                judge=judge,
                batch_size=batch_size,
                threshold=threshold,
                exact_uf=exact_uf,
                resume_state=resume_state,
                on_batch_complete=_on_batch,
            )
            by_type[kex_type] = metrics

        if ckptr:
            ckptr.mark_type_completed(kex_type, by_type[kex_type])

        pbar.update(1)

    pbar.close()

    if ckptr and not skip_types:
        ckptr.cleanup()

    return DedupReport(model_name=model_name, by_type=by_type)


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
