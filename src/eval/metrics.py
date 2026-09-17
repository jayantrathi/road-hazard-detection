from __future__ import annotations
import numpy as np


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve. 0.5 = random, 1.0 = perfect ranking.
    Computed via the rank-sum (Mann-Whitney U) identity, no sklearn needed."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Need both anomaly and normal samples for AUROC")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _assign_tie_ranks(scores, ranks)
    sum_pos = ranks[labels == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _assign_tie_ranks(scores, ranks):
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1


def aupr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision).
    More informative than AUROC when anomalies are rare."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    precision = tp / (tp + fp)
    total_pos = labels.sum()
    if total_pos == 0:
        raise ValueError("No positive (anomaly) samples")
    recall = tp / total_pos
    # integrate precision over recall increments
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_r)
        prev_r = r
    return float(ap)


def fpr_at_recall(scores: np.ndarray, labels: np.ndarray, target_recall: float = 0.95) -> float:
    """False-positive rate when the threshold is set to catch `target_recall`
    of anomalies. This is the operating-point number that actually matters for
    deployment: 'to catch 95% of corner cases, how often do we false-alarm?'"""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    # threshold = the score at which we've recalled target_recall of anomalies
    thresh = np.quantile(pos, 1 - target_recall)
    fp = (neg >= thresh).sum()
    return float(fp / len(neg))


def summarize(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "AUROC": auroc(scores, labels),
        "AUPR": aupr(scores, labels),
        "FPR@95": fpr_at_recall(scores, labels, 0.95),
    }


if __name__ == "__main__":
    # quick self-check on synthetic data
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, 1000)
    anom = rng.normal(2.5, 1, 100)
    scores = np.concatenate([normal, anom])
    labels = np.concatenate([np.zeros(1000), np.ones(100)])
    print("Synthetic sanity check (should be a decent but imperfect detector):")
    for k, v in summarize(scores, labels).items():
        print(f"  {k}: {v:.3f}")
