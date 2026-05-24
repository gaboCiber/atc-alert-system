from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

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
    char_f1: float
    char_precision: float
    char_recall: float
    word_f1: float
    word_precision: float
    word_recall: float
    rouge_l: float
    token_overlap_ratio: float
    fuzzy_match_ratio: float
    chunk_levenshtein: List[float]


@dataclass
class PageMetrics:
    page: int
    structural: StructuralMetrics
    content: ContentMetrics


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).lower()


def _get_words(text: str) -> List[str]:
    import re
    return re.findall(r"\b\w+\b", _normalize_text(text))


def _lcs_length(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def _best_chunk_match_ratio(
    chunks_a: List[Chunk], chunks_b: List[Chunk], scorer
) -> Tuple[float, List[float]]:
    if not chunks_a or not chunks_b:
        return 0.0, []

    text_a = [c.text for c in chunks_a]
    text_b = [c.text for c in chunks_b]

    scores = []
    for ta in text_a:
        best = max(scorer(ta, tb) for tb in text_b)
        scores.append(best)

    return np.mean(scores), scores


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


def compute_structural_metrics(pred: PageChunks, gt: PageChunks) -> StructuralMetrics:
    pred_chunks = pred.chunks
    gt_chunks = gt.chunks

    pcount = len(pred_chunks)
    gcount = len(gt_chunks)
    ccount_error = abs(pcount - gcount)
    ccount_acc = 1.0 - ccount_error / max(gcount, 1)

    edges, unmatched_pred, unmatched_gt = _match_bipartite(pred_chunks, gt_chunks)

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
        pred_texts = [c.text for c in pred_chunks[: pi + 1]]
        gt_texts = [c.text for c in gt_chunks[: gi + 1]]
        boundary_errors.append(abs(len(pred_texts) - len(gt_texts)))

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


def compute_content_metrics(pred: PageChunks, gt: PageChunks) -> ContentMetrics:
    pred_chunks = pred.chunks
    gt_chunks = gt.chunks

    def char_metrics(p_texts: List[str], g_texts: List[str]) -> Tuple[float, float, float]:
        p_all = " ".join(p_texts)
        g_all = " ".join(g_texts)
        p_chars = _normalize_text(p_all)
        g_chars = _normalize_text(g_all)
        p_set = set(p_chars.split())
        g_set = set(g_chars.split())
        if not g_set:
            return 0.0, 0.0, 0.0
        inter = len(p_set & g_set)
        prec = inter / len(p_set) if p_set else 0.0
        rec = inter / len(g_set) if g_set else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return f1, prec, rec

    char_f1, char_prec, char_rec = char_metrics(
        [c.text for c in pred_chunks], [c.text for c in gt_chunks]
    )

    def word_metrics(p_texts: List[str], g_texts: List[str]) -> Tuple[float, float, float]:
        p_words = _get_words(" ".join(p_texts))
        g_words = _get_words(" ".join(g_texts))
        if not g_words:
            return 0.0, 0.0, 0.0
        inter_len = _lcs_length(p_words, g_words)
        prec = inter_len / len(p_words) if p_words else 0.0
        rec = inter_len / len(g_words) if g_words else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return f1, prec, rec

    word_f1, word_prec, word_rec = word_metrics(
        [c.text for c in pred_chunks], [c.text for c in gt_chunks]
    )

    pred_words_list = _get_words(" ".join(c.text for c in pred_chunks))
    gt_words_list = _get_words(" ".join(c.text for c in gt_chunks))
    r_lcs = _lcs_length(pred_words_list, gt_words_list)
    denom = max(len(pred_words_list), len(gt_words_list))
    rouge_l = r_lcs / denom if denom > 0 else 0.0

    token_over = _lcs_length(pred_words_list, gt_words_list) / max(len(gt_words_list), 1)

    fuzzy_score, fuzzy_scores = _best_chunk_match_ratio(
        pred_chunks, gt_chunks, lambda a, b: fuzz.ratio(a, b)
    )

    lev_scores = []
    for pc in pred_chunks:
        best_lev = min(
            Levenshtein.distance(pc.text, gc.text) for gc in gt_chunks
        ) if gt_chunks else 0
        max_len = max(
            max(len(pc.text), len(gc.text)) for gc in gt_chunks
        ) if gt_chunks else 0
        norm_lev = 1.0 - best_lev / max_len if max_len > 0 else 0.0
        lev_scores.append(norm_lev)

    return ContentMetrics(
        char_f1=char_f1,
        char_precision=char_prec,
        char_recall=char_rec,
        word_f1=word_f1,
        word_precision=word_prec,
        word_recall=word_rec,
        rouge_l=rouge_l,
        token_overlap_ratio=token_over,
        fuzzy_match_ratio=fuzzy_score / 100.0,
        chunk_levenshtein=lev_scores,
    )


def compute_page_metrics(page: int, pred: PageChunks, gt: PageChunks) -> PageMetrics:
    structural = compute_structural_metrics(pred, gt)
    content = compute_content_metrics(pred, gt)
    return PageMetrics(page=page, structural=structural, content=content)