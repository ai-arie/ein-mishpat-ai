"""KesherBench evaluation: every task × setting × split in configs/eval/kesher.yaml.

Corpus embeddings: scripts/encode_corpus.py -> artifacts/emb/kesher/<model>/<corpus>_<lang>/.
Model "bm25" runs the BM25 baseline instead (he->he and en->en only).
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import polars as pl

from ein_mishpat_ai.eval.bm25 import bm25_topk
from ein_mishpat_ai.eval.encode import load_embeddings
from ein_mishpat_ai.eval.metrics import qrels_metrics, topk
from ein_mishpat_ai.eval.record import write_run
from ein_mishpat_ai.utils.io import load_yaml, resolve
from ein_mishpat_ai.utils.seed import set_seed

SCORE_SETTINGS = [  # the 8 agreed settings of the KesherBench score
    "K1_halacha_talmud/he2he",
    "K1_halacha_talmud/en2he",
    "K2a_commentary_torah/he2he",
    "K2a_commentary_torah/en2he",
    "K2a_commentary_torah/en2en",
    "K2b_commentary_talmud/he2he",
    "K4b_yerushalmi_bavli/he2he",
    "K4b_yerushalmi_bavli/en2he",
]


def kesher_score(metrics: dict[str, float]) -> float:
    return sum(metrics[f"{s}/ndcg_at_10"] for s in SCORE_SETTINGS) / len(SCORE_SETTINGS)


def topic_hit10(metrics: dict[str, float]) -> float:
    return (
        metrics["K5_topic_sources/he2he/hit_at_10"] + metrics["K5_topic_sources/en2he/hit_at_10"]
    ) / 2


def flagship_choice(short_run: dict[str, float], longer_run: dict[str, float]) -> bool:
    """Rule fixed on 2026-11-07, before the longer run: use it only if its dev score is
    higher and its topic search (hit@10, mean of Hebrew and English) is not lower."""
    return kesher_score(longer_run) > kesher_score(short_run) and topic_hit10(
        longer_run
    ) >= topic_hit10(short_run)


def evaluate(
    model: str,
    cfg_path: str = "configs/eval/kesher.yaml",
    split: str | None = None,
    only: list[str] | None = None,
) -> dict[str, Any]:
    """`only`: "task/setting" names, for a model that covers just those (torah-embed: en->en)."""
    cfg = load_yaml(cfg_path)
    set_seed(cfg["seed"])
    split = split or cfg["split"]
    root = resolve(cfg["bench_dir"])
    enc = None
    if model != "bm25":
        from ein_mishpat_ai.eval.encoder import Encoder, model_spec

        enc = Encoder(model_spec(model, max_length=cfg["max_length"], batch_size=cfg["batch_size"]))
    metrics: dict[str, float] = {}
    counts: dict[str, int] = {}
    corpus_cache: dict[str, tuple[list[str], Any]] = {}
    t0 = time.time()
    for task, instruction in cfg["tasks"].items():
        meta = json.loads((root / task / "task.json").read_text())
        qrels = pl.read_parquet(root / task / f"qrels_{split}.parquet")
        for setting in meta["settings"]:
            qf = root / task / f"queries_{setting}_{split}.parquet"
            if not qf.exists() or (only and f"{task}/{setting}" not in only):
                continue
            qlang, clang = setting[:2], setting[3:]
            if model == "bm25" and qlang != clang:
                continue
            cname = f"{meta['corpus']}_{clang}"
            if cname not in corpus_cache:
                if model == "bm25":
                    c = pl.read_parquet(
                        root / "corpora" / f"{cname}.parquet", columns=["id", "text"]
                    )
                    corpus_cache[cname] = (c["id"].to_list(), c["text"].to_list())
                else:
                    ids, emb = load_embeddings(resolve(cfg["emb_dir"]) / model / cname)
                    n = pl.read_parquet(
                        root / "corpora" / f"{cname}.parquet", columns=["id"]
                    ).height
                    if len(ids) != n:
                        raise RuntimeError(f"{model}/{cname}: {len(ids)} of {n} embedded")
                    corpus_cache[cname] = (ids, emb.astype(np.float32))
            ids, cdata = corpus_cache[cname]
            col = {c: i for i, c in enumerate(ids)}
            q = pl.read_parquet(qf)
            pos: dict[str, set[int]] = {}
            for qid, cid in qrels.select("query_id", "corpus_id").iter_rows():
                if cid in col:
                    pos.setdefault(qid, set()).add(col[cid])
            q = q.filter(pl.col("id").is_in(list(pos)))
            if not q.height:
                continue
            k = max(cfg["k_values"])
            if model == "bm25":
                idx = bm25_topk(cdata, q["text"].to_list(), qlang, k)
            else:
                qe = enc.encode(q["text"].to_list(), prefix=enc.query_prefix(instruction))
                _, idx = topk(qe, cdata, k)
            m = qrels_metrics(
                idx, [pos[i] for i in q["id"].to_list()], cfg["k_values"], cfg["mrr_k"]
            )
            metrics.update({f"{task}/{setting}/{kk}": v for kk, v in m.items()})
            if (
                split == "test"
            ):  # per-query results for the bootstrap, breakdowns and failure gallery
                top = resolve(cfg["out_dir"]) / split / model / "topk"
                top.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    top / f"{task}__{setting}.npz",
                    query_ids=np.array(q["id"].to_list()),
                    corpus_ids=np.array(ids)[idx],
                )
            counts[f"{task}/{setting}"] = q.height
    out = resolve(cfg["out_dir"]) / split / model
    return write_run(
        out,
        experiment=f"kesher_{split}",
        run_name=model,
        config={**cfg, "model": model, "split": split},
        metrics=metrics,
        extra={"n_queries": counts, "seconds": round(time.time() - t0, 1)},
    )
