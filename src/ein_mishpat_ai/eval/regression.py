"""General-domain regression check: do our models lose quality outside Jewish texts?

Public retrieval tasks (configs/eval/regression.yaml), normalised to corpus / queries / qrels Parquet
by `prepare`. `evaluate` encodes both sides with the model and scores with the same metrics code as
every other benchmark. Per-query nDCG@10 is kept for the paired bootstrap.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ein_mishpat_ai.eval.encoder import Encoder, model_spec
from ein_mishpat_ai.eval.metrics import qrels_metrics, topk
from ein_mishpat_ai.eval.record import write_run
from ein_mishpat_ai.utils.io import load_yaml, resolve
from ein_mishpat_ai.utils.seed import set_seed

CONFIG = "configs/eval/regression.yaml"


def prepare(cfg_path: str = CONFIG) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    cfg = load_yaml(cfg_path)
    stats = {}
    for name, t in cfg["tasks"].items():
        out = resolve(cfg["data_dir"]) / name
        out.mkdir(parents=True, exist_ok=True)
        kw = {"repo_type": "dataset", "revision": t["revision"]}
        if t["format"] == "webfaq":
            corpus, queries, qrels = (
                pl.read_parquet(
                    hf_hub_download(t["repo"], f"{t['lang']}-{s}/test-00000-of-00001.parquet", **kw)
                )
                for s in ("corpus", "queries", "qrels")
            )
        else:
            corpus = pl.read_ndjson(hf_hub_download(t["repo"], "corpus.jsonl", **kw)).rename(
                {"_id": "id"}
            )
            queries = pl.read_ndjson(hf_hub_download(t["repo"], "queries.jsonl", **kw)).rename(
                {"_id": "id"}
            )
            qrels = pl.read_csv(hf_hub_download(t["repo"], "qrels/test.tsv", **kw), separator="\t")
        qrels = qrels.rename({c: c.replace("-", "_") for c in qrels.columns})
        as_str = pl.col("id").cast(pl.String)
        corpus = corpus.with_columns(as_str, pl.col("title").fill_null(""))
        queries = queries.with_columns(as_str)
        qrels = qrels.with_columns(
            pl.col("query_id").cast(pl.String), pl.col("corpus_id").cast(pl.String)
        ).filter(pl.col("score") > 0)
        queries = queries.filter(pl.col("id").is_in(qrels["query_id"].implode()))
        corpus.select("id", "title", "text").write_parquet(out / "corpus.parquet")
        queries.select("id", "text").write_parquet(out / "queries.parquet")
        qrels.select("query_id", "corpus_id").write_parquet(out / "qrels.parquet")
        stats[name] = {"corpus": corpus.height, "queries": queries.height, "qrels": qrels.height}
    return stats


def evaluate(model: str, cfg_path: str = CONFIG) -> dict[str, Any]:
    cfg = load_yaml(cfg_path)
    set_seed(cfg["seed"])
    enc = Encoder(model_spec(model, max_length=cfg["max_length"], batch_size=cfg["batch_size"]))
    out = resolve(cfg["out_dir"]) / model
    out.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name, t in cfg["tasks"].items():
        d = resolve(cfg["data_dir"]) / name
        c = pl.read_parquet(d / "corpus.parquet")
        q = pl.read_parquet(d / "queries.parquet")
        col = {x: i for i, x in enumerate(c["id"].to_list())}
        pos: dict[str, set[int]] = {}
        for qid, cid in pl.read_parquet(d / "qrels.parquet").iter_rows():
            if cid in col:
                pos.setdefault(qid, set()).add(col[cid])
        q = q.filter(pl.col("id").is_in(list(pos)))
        docs = [f"{a} {b}".strip() for a, b in zip(c["title"], c["text"], strict=True)]
        ce = enc.encode(docs, prefix=enc.spec.doc_prompt)
        qe = enc.encode(q["text"].to_list(), prefix=enc.query_prefix(t["instruction"]))
        _, idx = topk(qe, ce, max(cfg["k_values"]))
        positives = [pos[i] for i in q["id"].to_list()]
        m = qrels_metrics(idx, positives, cfg["k_values"], cfg["mrr_k"])
        metrics.update({f"{name}/{k}": v for k, v in m.items()})
        counts[name] = q.height
        per_query = [
            qrels_metrics(idx[i : i + 1], [positives[i]], [10], cfg["mrr_k"])["ndcg_at_10"]
            for i in range(len(positives))
        ]
        np.save(out / f"{name}_ndcg10.npy", np.array(per_query, dtype=np.float32))
    return write_run(
        out,
        experiment=cfg["name"],
        run_name=model,
        config={**cfg, "model": model},
        metrics=metrics,
        extra={"n_queries": counts, "params": enc.n_params},
    )
