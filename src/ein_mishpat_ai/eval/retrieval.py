"""Link-retrieval evaluation on a task dir with corpus.parquet, queries_<lang>.parquet, qrels.parquet.

Corpus embeddings come from scripts/encode_corpus.py (documents: no prompt); queries are encoded
here with the task instruction (prompt_style from configs/eval/models.yaml).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import polars as pl

from ein_mishpat_ai.eval.encode import load_embeddings
from ein_mishpat_ai.eval.encoder import Encoder, model_spec
from ein_mishpat_ai.eval.metrics import qrels_metrics, topk
from ein_mishpat_ai.eval.record import write_run
from ein_mishpat_ai.utils.io import load_yaml, resolve
from ein_mishpat_ai.utils.seed import set_seed


def evaluate(model: str, cfg_path: str) -> dict[str, Any]:
    cfg = load_yaml(cfg_path)
    set_seed(cfg["seed"])
    task = resolve(cfg["task_dir"])
    ids, emb = load_embeddings(resolve(cfg["emb_dir"]) / model / "corpus")
    n_corpus = pl.read_parquet(task / "corpus.parquet", columns=["id"]).height
    if len(ids) != n_corpus:
        raise RuntimeError(f"corpus embeddings incomplete: {len(ids)} of {n_corpus}")
    col = {c: i for i, c in enumerate(ids)}
    qrels = pl.read_parquet(task / "qrels.parquet")
    pos_by_q: dict[str, set[int]] = {}
    for q, c in qrels.iter_rows():
        pos_by_q.setdefault(q, set()).add(col[c])

    enc = Encoder(model_spec(model, max_length=cfg["max_length"], batch_size=cfg["batch_size"]))
    prefix = enc.query_prefix(cfg["instruction"])
    corpus = emb.astype(np.float32)
    metrics: dict[str, float] = {}
    counts: dict[str, int] = {}
    t0 = time.time()
    for lang in ("he", "en"):
        qf = task / f"queries_{lang}.parquet"
        q = pl.read_parquet(qf) if qf.exists() else pl.DataFrame()
        q = q.filter(pl.col("id").is_in(list(pos_by_q))) if q.height else q
        if not q.height:
            continue
        qe = enc.encode(q["text"].to_list(), prefix=prefix)
        _, idx = topk(qe, corpus, max(cfg["k_values"]))
        m = qrels_metrics(
            idx, [pos_by_q[i] for i in q["id"].to_list()], cfg["k_values"], cfg["mrr_k"]
        )
        setting = f"{lang}2he"
        metrics.update({f"{setting}/{k}": v for k, v in m.items()})
        counts[setting] = q.height
    out = resolve(cfg["out_dir"]) / model
    config = {**cfg, "model": model, "query_prefix": prefix}
    return write_run(
        out,
        experiment=cfg["name"],
        run_name=model,
        config=config,
        metrics=metrics,
        extra={
            "n_queries": counts,
            "n_corpus": n_corpus,
            "query_seconds": round(time.time() - t0, 1),
        },
    )
