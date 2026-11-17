"""Retrieval metrics.

`space_*` functions reproduce Sefaria's evaluation.py exactly (same argsort, same RNG draws), so
numbers are comparable with the published leaderboard. `qrels_metrics` is the general version used
for link benchmarks (several positives per query, corpus != query set).
"""

from __future__ import annotations

import numpy as np


def space_true_ranks(sim: np.ndarray) -> np.ndarray:
    """0-based rank of passage i for query i, using the same argsort as evaluation.py."""
    rankings = np.argsort(-sim, axis=1)
    return np.argmax(rankings == np.arange(sim.shape[0])[:, None], axis=1)


def space_retrieval_metrics(sim: np.ndarray, k_values: list[int]) -> dict[str, float]:
    ranks = space_true_ranks(sim)
    out = {f"recall_at_{k}": float(np.mean(ranks < k)) for k in k_values}
    out["mrr"] = float(np.mean(1.0 / (ranks + 1)))
    return out


def space_bitext_accuracy(
    sim: np.ndarray, num_negatives: int = 10, seed: int = 42
) -> tuple[float, float, float]:
    """(accuracy, avg true similarity, avg random similarity), identical RNG use to evaluation.py."""
    n = sim.shape[0]
    true = np.diag(sim)
    rng = np.random.default_rng(seed)
    correct = total = 0
    random_sims = []
    for i in range(n):
        neg = rng.choice(
            [j for j in range(n) if j != i], size=min(num_negatives, n - 1), replace=False
        )
        s = sim[i, neg]
        random_sims.append(s)  # keep the matrix dtype: the reference averages float32 scalars
        correct += int(np.sum(true[i] > s))
        total += len(neg)
    acc = correct / total if total else 0.0
    avg_random = float(np.mean(np.concatenate(random_sims))) if random_sims else 0.0
    return acc, float(np.mean(true)), avg_random


def topk(q: np.ndarray, c: np.ndarray, k: int, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Exact inner-product top-k (scores, indices), best first. Uses torch on GPU when available."""
    k = min(k, c.shape[0])
    try:
        import torch

        if torch.cuda.is_available():
            ct = torch.from_numpy(c).cuda().half()
            scores, idx = [], []
            for i in range(0, q.shape[0], chunk):
                s = torch.from_numpy(q[i : i + chunk]).cuda().half() @ ct.T
                v, j = torch.topk(s.float(), k, dim=1)
                scores.append(v.cpu().numpy())
                idx.append(j.cpu().numpy())
            return np.vstack(scores), np.vstack(idx)
    except ImportError:
        pass
    scores, idx = [], []
    for i in range(0, q.shape[0], chunk):
        s = q[i : i + chunk] @ c.T
        j = np.argpartition(-s, k - 1, axis=1)[:, :k]
        v = np.take_along_axis(s, j, axis=1)
        o = np.argsort(-v, axis=1)
        scores.append(np.take_along_axis(v, o, axis=1))
        idx.append(np.take_along_axis(j, o, axis=1))
    return np.vstack(scores), np.vstack(idx)


def qrels_metrics(
    retrieved: np.ndarray, positives: list[set[int]], k_values: list[int], mrr_k: int = 10
) -> dict[str, float]:
    """Recall@k = fraction of a query's positives in its top k, averaged over queries (macro).
    Also hit@k (at least one positive in top k), nDCG@10 (binary gains) and MRR@mrr_k."""
    out: dict[str, float] = {}
    n = len(positives)
    for k in k_values:
        rec = hit = 0.0
        for row, pos in zip(retrieved[:, :k], positives, strict=True):
            found = len(pos.intersection(row.tolist()))
            rec += found / len(pos)
            hit += found > 0
        out[f"recall_at_{k}"] = rec / n
        out[f"hit_at_{k}"] = hit / n
    nd = 0.0
    for row, pos in zip(retrieved[:, :10], positives, strict=True):
        dcg = sum(1.0 / np.log2(r + 2) for r, j in enumerate(row.tolist()) if j in pos)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(pos), 10)))
        nd += dcg / idcg
    out["ndcg_at_10"] = nd / n
    rr = 0.0
    for row, pos in zip(retrieved[:, :mrr_k], positives, strict=True):
        for r, j in enumerate(row.tolist()):
            if j in pos:
                rr += 1.0 / (r + 1)
                break
    out[f"mrr_at_{mrr_k}"] = rr / n
    return out
