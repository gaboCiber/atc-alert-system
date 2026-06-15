from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from rapidfuzz import fuzz

from loader import PageChunks, Chunk


@dataclass
class StructuralMetrics:
    chunk_count: int
    gt_chunk_count: int
    chunk_count_error: int
    chunk_count_accuracy: float
    boundary_f1: float
    boundary_precision: float
    boundary_recall: float
    boundary_avg_error: float
    chunk_boundary_errors: List[int]


@dataclass
class ContentMetrics:
    matched_content_precision: float = 0.0
    matched_content_recall: float = 0.0
    matched_content_f1: float = 0.0


@dataclass
class PageMetrics:
    page: int
    structural: StructuralMetrics
    content: ContentMetrics


def _match_bipartite(
    chunks_a: List[Chunk], chunks_b: List[Chunk], threshold: float = 0.5
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    n, m = len(chunks_a), len(chunks_b)
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))

    texts_a = [c.text for c in chunks_a]
    texts_b = [c.text for c in chunks_b]

    score_matrix = np.zeros((n, m))
    for i, ta in enumerate(texts_a):
        for j, tb in enumerate(texts_b):
            score_matrix[i, j] = fuzz.ratio(ta, tb) / 100.0

    matched_a, matched_b = set(), set()
    edges = []

    sorted_pairs = sorted(
        ((i, j, score_matrix[i, j]) for i in range(n) for j in range(m)),
        key=lambda x: -x[2],
    )

    for i, j, score in sorted_pairs:
        if score < threshold:
            break
        if i not in matched_a and j not in matched_b:
            edges.append((i, j))
            matched_a.add(i)
            matched_b.add(j)

    unmatched_a = [i for i in range(n) if i not in matched_a]
    unmatched_b = [j for j in range(m) if j not in matched_b]

    return edges, unmatched_a, unmatched_b


def compute_structural_metrics(
    pred: PageChunks, gt: PageChunks,
    edges: List[Tuple[int, int]],
    unmatched_pred: List[int],
    unmatched_gt: List[int],
) -> StructuralMetrics:
    pred_chunks = pred.chunks
    gt_chunks = gt.chunks

    pcount = len(pred_chunks)
    gcount = len(gt_chunks)
    ccount_error = abs(pcount - gcount)
    ccount_acc = 1.0 - ccount_error / max(gcount, 1)

    if not gt_chunks:
        return StructuralMetrics(
            chunk_count=pcount,
            gt_chunk_count=gcount,
            chunk_count_error=ccount_error,
            chunk_count_accuracy=ccount_acc,
            boundary_f1=0.0,
            boundary_precision=0.0,
            boundary_recall=0.0,
            boundary_avg_error=0.0,
            chunk_boundary_errors=[],
        )

    boundary_errors = []
    for pi, gi in edges:
        boundary_errors.append(abs((pi + 1) - (gi + 1)))

    tp = len(edges)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    avg_err = np.mean(boundary_errors) if boundary_errors else 0.0

    return StructuralMetrics(
        chunk_count=pcount,
        gt_chunk_count=gcount,
        chunk_count_error=ccount_error,
        chunk_count_accuracy=ccount_acc,
        boundary_f1=f1,
        boundary_precision=prec,
        boundary_recall=rec,
        boundary_avg_error=avg_err,
        chunk_boundary_errors=boundary_errors,
    )


def compute_matched_content_metrics(
    pred: PageChunks, gt: PageChunks,
    edges: List[Tuple[int, int]],
    unmatched_pred: List[int],
    unmatched_gt: List[int],
) -> ContentMetrics:
    total_pred = len(pred.chunks)
    total_gt = len(gt.chunks)

    if total_pred == 0 and total_gt == 0:
        return ContentMetrics(
            matched_content_precision=1.0,
            matched_content_recall=1.0,
            matched_content_f1=1.0,
        )

    if total_pred == 0 or total_gt == 0:
        return ContentMetrics(
            matched_content_precision=0.0,
            matched_content_recall=0.0,
            matched_content_f1=0.0,
        )

    scores = []
    for pi, gi in edges:
        scores.append(fuzz.ratio(pred.chunks[pi].text, gt.chunks[gi].text) / 100.0)

    sum_scores = sum(scores)
    prec = sum_scores / total_pred
    rec = sum_scores / total_gt
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return ContentMetrics(
        matched_content_precision=prec,
        matched_content_recall=rec,
        matched_content_f1=f1,
    )


def compute_boundary_integrity_document(
    all_chunks: List[Chunk],
    sentences_gt: Optional[List[str]] = None,
) -> Tuple[float, Dict]:
    if not all_chunks:
        return 1.0, {"num_chunks": 0, "note": "no chunks"}

    norm_chunks = [" ".join(c.text.split()) for c in all_chunks]
    full_text = " ".join(norm_chunks)

    if sentences_gt is not None:
        norm_sentences = [" ".join(s.split()) for s in sentences_gt]
        sent_ends = []
        pos = 0
        for s in norm_sentences:
            pos += len(s)
            sent_ends.append(pos)
            pos += 1
    else:
        try:
            import nltk
            nltk_sentences = nltk.sent_tokenize(full_text)
        except (ImportError, LookupError):
            import re
            nltk_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_text) if s.strip()]

        sent_ends = []
        pos = 0
        for s in nltk_sentences:
            pos += len(s)
            sent_ends.append(pos)
            pos += 1

    chunk_ends = []
    pos = 0
    for c in norm_chunks:
        pos += len(c)
        chunk_ends.append(pos)
        pos += 1

    good_ends = []
    bad_ends = []
    for ce in chunk_ends:
        matched = False
        for se in sent_ends:
            if ce == se:
                good_ends.append(ce)
                matched = True
                break
        if not matched:
            bad_ends.append(ce)

    score = len(good_ends) / len(chunk_ends) if chunk_ends else 1.0

    detail = {
        "num_chunks": len(all_chunks),
        "full_text": full_text,
        "sentence_ends": sent_ends,
        "chunk_ends": chunk_ends,
        "matched_ends": good_ends,
        "unmatched_ends": bad_ends,
        "score": score,
    }
    return score, detail


def compute_boundary_integrity(pred: PageChunks) -> Tuple[float, Dict]:
    chunks = pred.chunks
    if not chunks:
        return 1.0, {"num_chunks": 0, "note": "no chunks"}

    norm_chunks = [" ".join(c.text.split()) for c in chunks]
    full_text = " ".join(norm_chunks)

    try:
        import nltk
        sentences = nltk.sent_tokenize(full_text)
    except (ImportError, LookupError):
        import re
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_text) if s.strip()]

    sent_spans = []
    pos = 0
    for s in sentences:
        sent_spans.append((pos, pos + len(s)))
        pos += len(s) + 1

    sent_ends = [end for _, end in sent_spans]

    chunk_ends = []
    pos = 0
    for i, c in enumerate(norm_chunks):
        pos += len(c)
        chunk_ends.append(pos)
        pos += 1

    good_ends = []
    bad_ends = []
    for ce in chunk_ends:
        matched = False
        for start, end in sent_spans:
            if ce == end:
                good_ends.append(ce)
                matched = True
                break
        if not matched:
            bad_ends.append(ce)

    score = len(good_ends) / len(chunk_ends) if chunk_ends else 1.0

    detail = {
        "num_chunks": len(chunks),
        "full_text": full_text,
        "sentence_ends": sent_ends,
        "chunk_ends": chunk_ends,
        "matched_ends": good_ends,
        "unmatched_ends": bad_ends,
        "score": score,
    }
    return score, detail


def compute_page_metrics(page: int, pred: PageChunks, gt: PageChunks) -> PageMetrics:
    edges, unmatched_pred, unmatched_gt = _match_bipartite(pred.chunks, gt.chunks)
    structural = compute_structural_metrics(pred, gt, edges, unmatched_pred, unmatched_gt)
    content = compute_matched_content_metrics(pred, gt, edges, unmatched_pred, unmatched_gt)
    return PageMetrics(page=page, structural=structural, content=content)