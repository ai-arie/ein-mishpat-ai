"""Our Space-protocol metrics must equal Sefaria's evaluation.py (logic copied literally below from
data/external/sefaria_space/evaluation.py at the pinned revision; the downloaded file is never run)."""

import numpy as np

from ein_mishpat_ai.eval import metrics as m


def _ref_retrieval(sim, k_values=(1, 5, 10)):
    n = sim.shape[0]
    rankings = np.argsort(-sim, axis=1)
    true_ranks = np.zeros(n, dtype=int)
    for i in range(n):
        true_ranks[i] = np.where(rankings[i] == i)[0][0]
    out = {f"recall_at_{k}": float(np.mean(true_ranks < k)) for k in k_values}
    out["mrr"] = float(np.mean(1.0 / (true_ranks + 1)))
    return out


def _ref_bitext(sim, num_negatives=10):
    n = sim.shape[0]
    true_similarities = np.diag(sim)
    correct = total = 0
    random_sims = []
    rng = np.random.default_rng(42)
    for i in range(n):
        true_sim = true_similarities[i]
        neg_indices = rng.choice(
            [j for j in range(n) if j != i], size=min(num_negatives, n - 1), replace=False
        )
        for j in neg_indices:
            neg_sim = sim[i, j]
            random_sims.append(neg_sim)
            if true_sim > neg_sim:
                correct += 1
            total += 1
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, float(np.mean(true_similarities)), float(np.mean(random_sims))


def _sim(n=300, d=16, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    y = x + rng.standard_normal((n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return (x @ y.T).astype(np.float32)


def test_retrieval_matches_reference():
    s = _sim()
    assert m.space_retrieval_metrics(s, [1, 5, 10]) == _ref_retrieval(s)


def test_bitext_matches_reference():
    s = _sim()
    assert m.space_bitext_accuracy(s, 10, 42) == _ref_bitext(s)


def test_qrels_metrics():
    retrieved = np.array([[3, 1, 2], [0, 2, 1]])
    pos = [{1, 9}, {5}]
    out = m.qrels_metrics(retrieved, pos, [1, 3], mrr_k=3)
    assert out["recall_at_1"] == 0.0
    assert out["recall_at_3"] == 0.25
    assert out["hit_at_3"] == 0.5
    assert out["mrr_at_3"] == 0.25
    # query 1: positive at rank 2 of 2 positives -> dcg 1/log2(3), idcg 1 + 1/log2(3)
    assert abs(out["ndcg_at_10"] - 0.5 * (1 / np.log2(3)) / (1 + 1 / np.log2(3))) < 1e-9


def test_topk_cpu_matches_bruteforce():
    rng = np.random.default_rng(1)
    q, c = (
        rng.standard_normal((20, 8)).astype(np.float32),
        rng.standard_normal((50, 8)).astype(np.float32),
    )
    _, idx = m.topk(q, c, 5)
    expect = np.argsort(-(q @ c.T), axis=1)[:, :5]
    assert (idx == expect).mean() > 0.95  # GPU path uses fp16 scores


def test_query_prefix_styles():
    from ein_mishpat_ai.eval.encoder import Encoder, model_spec

    enc = Encoder.__new__(Encoder)  # no model load: only the prompt logic
    enc.spec = model_spec("e5-large")
    assert enc.query_prefix("any task") == "query: " and enc.spec.doc_prompt == "passage: "
    enc.spec = model_spec("f2llm-1.7b")
    assert enc.query_prefix("Find X") == "Instruct: Find X\nQuery: "
    enc.spec = model_spec("bge-m3")
    assert enc.query_prefix("Find X") == ""
