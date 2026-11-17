"""M4 training examples from the M3 outputs. usage: scripts/train.py mix

One row per (query, direction): task, query_ref, qlang, tlang, positives, negatives (mined), and the
teacher's scores for them (distillation; none for synthetic text). Texts live once in texts.parquet,
keyed by (lang, ref); lang "syn" = D3a synthetic English (training only, project rule 4 (README)).

- D1 bitext he→en and en→he, filtered by `d1_filter`, minus a held-out slice of
  whole works (`dev_d1.parquet`, model selection only).
- D2 link tasks: Hebrew or English query → Hebrew target (the pool was mined in Hebrew).
- D3a: Hebrew source ↔ synthetic English, filtered by `d3a_filter`; synthetic English → Hebrew has
  the teacher's mined Hebrew negatives and scores (filters.d3a_mine), Hebrew → English in-batch only.
- D3b (optional, mix config `d3b`): synthetic Hebrew / English search query → Hebrew passage, kept
  if the teacher ranks the passage in its top `keep_rank` (filters.d3b_mine), with mined negatives.
  Query texts are stored as lang synq_he / synq_en (synthetic, training only).
Every ref comes from the teacher tables or the D3a selection, which hold training refs only
(tests/test_leakage.py checks the written mix as well).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import polars as pl

from ein_mishpat_ai.utils.io import load_yaml, resolve
from ein_mishpat_ai.utils.log import get_logger

log = get_logger("train", "train.log")
LANGS = ("he", "en")


def _held_out_works(titles: list[str], share: float, seed: int) -> set[str]:
    def h(t: str) -> float:
        return int(hashlib.sha1(f"{seed}:{t}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    return {t for t in set(titles) if h(t) < share}


def build(config: str = "configs/train/mix_d3b_all.yaml") -> dict[str, Any]:
    mc = load_yaml(config)
    tc = load_yaml("configs/teacher.yaml")
    T = resolve(tc["out_dir"])
    out = resolve(mc["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    # texts: (lang, ref) -> text, exactly the forms the teacher encoded
    texts = [
        pl.read_parquet(T / f"docs_{lang}.parquet").select(pl.lit(lang).alias("lang"), "id", "text")
        for lang in LANGS
    ]
    texts += [
        pl.read_parquet(f).select(pl.lit(f.stem.rsplit("_", 1)[1]).alias("lang"), "id", "text")
        for f in T.glob("q_*.parquet")
    ]

    # held-out slice of D1 (whole works, liturgy excluded) for model selection
    title = (
        pl.read_parquet(T / "queries.parquet")
        .select("query_ref", "title")
        .unique("query_ref")
        .rename({"query_ref": "ref"})
    )
    align = pl.read_parquet(T / "d1_align.parquet").join(title, on="ref", how="left")
    lit = pl.col("title").str.contains(tc["d3a_filter"]["liturgy_pattern"])
    held = _held_out_works(
        align.filter(~lit)["title"].drop_nulls().to_list(), mc["dev_d1"]["work_share"], mc["seed"]
    )
    dev = align.filter(pl.col("title").is_in(list(held)) & pl.col("keep"))
    dev = dev.sample(min(mc["dev_d1"]["max_pairs"], dev.height), seed=mc["seed"])
    held_refs = set(align.filter(pl.col("title").is_in(list(held)))["ref"].to_list())
    keep_d1 = set(align.filter(pl.col("keep"))["ref"].to_list()) - held_refs

    rows = []
    for task, t in tc["tasks"].items():
        neg = pl.read_parquet(T / "negatives" / f"{task}.parquet")
        pairs = [(t["qlang"], t["tlang"])] if t["source"] == "d1" else [(ql, "he") for ql in LANGS]
        for ql, tl in pairs:
            col = f"s_{ql}_{tl}"
            if col not in neg.columns:
                continue
            d = neg.filter(~pl.col("query_ref").is_in(list(held_refs)))
            if t["source"] == "d1":
                d = d.filter(pl.col("query_ref").is_in(list(keep_d1)))
            d = d.select(
                pl.lit(task).alias("task"),
                "query_ref",
                pl.lit(ql).alias("qlang"),
                pl.lit(tl).alias("tlang"),
                pl.col("cands").list.head(pl.col("n_pos")).alias("positives"),
                pl.col("cands").list.slice(pl.col("n_pos")).alias("negatives"),
                pl.col(col).list.head(pl.col("n_pos")).cast(pl.List(pl.Float32)).alias("t_pos"),
                pl.col(col).list.slice(pl.col("n_pos")).cast(pl.List(pl.Float32)).alias("t_neg"),
                pl.lit(t["source"]).alias("group"),
            )
            rows.append(d)
    ex = pl.concat(rows)

    # D3a: only translations that passed the round trip filter (scored blocks only so far)
    from ein_mishpat_ai.gen.d3a import load as load_d3a

    rt = resolve(load_yaml("configs/gen.yaml")["d3a"]["out_dir"]) / "roundtrip"
    scores = [pl.read_parquet(f, columns=["ref", "keep"]) for f in rt.glob("score_*block*.parquet")]
    syn = load_d3a().filter(pl.col("translation").is_not_null())
    n_syn = syn.height
    redo = set(syn.filter(pl.col("redo"))["ref"].to_list())
    sc = pl.concat(scores).with_columns(pl.col("ref").is_in(list(redo)).alias("redo_score"))
    # a redone translation is judged by the score of its redo, the others by their block's score
    syn = syn.join(
        sc.group_by("ref").agg(pl.col("keep").sort_by("redo_score").last()), on="ref", how="inner"
    ).filter(pl.col("keep") & ~pl.col("ref").is_in(list(held_refs)))
    texts.append(
        syn.select(
            pl.lit("syn").alias("lang"),
            pl.col("ref").alias("id"),
            pl.col("translation").alias("text"),
        )
    )
    # mined Hebrew negatives for synthetic English → Hebrew (filters.d3a_mine), the redo's for a
    # redone translation; Hebrew → synthetic English keeps in-batch negatives only
    neg_files = sorted(rt.glob("neg_*block*.parquet"))
    if neg_files:
        negs_d3a = (
            pl.concat(
                [
                    pl.read_parquet(f).with_columns(
                        pl.lit(f.stem.startswith("neg_redo_")).alias("from_redo")
                    )
                    for f in neg_files
                ]
            )
            .filter(pl.col("from_redo") == pl.col("ref").is_in(list(redo)))
            .drop("from_redo")
        )
    else:
        negs_d3a = pl.DataFrame(
            schema={
                "ref": pl.String,
                "negatives": pl.List(pl.String),
                "t_pos": pl.List(pl.Float32),
                "t_neg": pl.List(pl.Float32),
            }
        )
    empty_s = pl.lit([], dtype=pl.List(pl.String))
    empty_f = pl.lit(None, dtype=pl.List(pl.Float32))
    for task, ql, tl in (("d3a_he2en", "he", "syn"), ("d3a_en2he", "syn", "he")):
        d = syn.select("ref")
        if tl == "he":
            d = d.join(negs_d3a, on="ref", how="left")
        else:
            d = d.with_columns(negatives=empty_s, t_pos=empty_f, t_neg=empty_f)
        ex = pl.concat(
            [
                ex,
                d.select(
                    pl.lit(task).alias("task"),
                    pl.col("ref").alias("query_ref"),
                    pl.lit(ql).alias("qlang"),
                    pl.lit(tl).alias("tlang"),
                    pl.concat_list(pl.col("ref")).alias("positives"),
                    "negatives",
                    "t_pos",
                    "t_neg",
                    pl.lit("d3a").alias("group"),
                ),
            ]
        )

    # D3b synthetic search queries → Hebrew passage (optional; separate mix dir, e.g. mix_d3b.yaml)
    n_d3b = {}
    if mc.get("d3b"):
        from ein_mishpat_ai.gen.d3b import load as load_d3b

        b = mc["d3b"]
        dirs = b.get("dirs") or [b["dir"]]  # several generation runs (pilot + rest), no shared refs
        q = pl.concat([load_d3b(x) for x in dirs])
        if q["ref"].n_unique() != q.height:
            raise ValueError(f"D3b runs share passages: {dirs}")
        for lang in LANGS:
            neg = pl.concat(
                [pl.read_parquet(resolve(x) / "roundtrip" / f"neg_{lang}.parquet") for x in dirs]
            ).filter(
                (pl.col("rank") > 0)
                & (pl.col("rank") <= b["keep_rank"])
                & ~pl.col("ref").is_in(list(held_refs))
            )
            n_d3b[lang] = {"queries": q.height, "kept": neg.height}
            texts.append(
                q.select(
                    pl.lit(f"synq_{lang}").alias("lang"),
                    pl.col("ref").alias("id"),
                    pl.col(f"q_{lang}").alias("text"),
                )
            )
            ex = pl.concat(
                [
                    ex,
                    neg.select(
                        pl.lit(f"d3b_{lang}2he").alias("task"),
                        pl.col("ref").alias("query_ref"),
                        pl.lit(f"synq_{lang}").alias("qlang"),
                        pl.lit("he").alias("tlang"),
                        pl.concat_list(pl.col("ref")).alias("positives"),
                        "negatives",
                        "t_pos",
                        "t_neg",
                        pl.lit("d3b").alias("group"),
                    ),
                ]
            )

    # held-out texts never appear in training, not even as negatives
    ex = ex.with_row_index("i")
    negs = (
        ex.select("i", "negatives", "t_neg")
        .with_columns(  # explode needs equal lengths: [] and null differ (polars 2)
            pl.col("negatives").fill_null(empty_s),
            pl.col("t_neg").fill_null(pl.lit([], dtype=pl.List(pl.Float32))),
        )
        .explode("negatives", "t_neg")
        .filter(pl.col("negatives").is_not_null() & ~pl.col("negatives").is_in(list(held_refs)))
        .group_by("i")
        .agg("negatives", "t_neg")
    )
    ex = ex.drop("negatives", "t_neg").join(negs, on="i", how="left").sort("i").drop("i")
    ex = ex.with_columns(
        pl.col("negatives").fill_null(empty_s), pl.col("t_neg").cast(pl.List(pl.Float32))
    ).select(
        "task", "query_ref", "qlang", "tlang", "positives", "negatives", "t_pos", "t_neg", "group"
    )

    tx = pl.concat(texts).unique(["lang", "id"]).rename({"id": "ref"})
    # D3a Hebrew sources must have their text (they are in docs_he, see M3 notes)
    have = set(zip(tx["lang"].to_list(), tx["ref"].to_list(), strict=True))
    ok = [
        (ql, q) in have and (tl, p[0]) in have
        for ql, tl, q, p in ex.select("qlang", "tlang", "query_ref", "positives").iter_rows()
    ]
    missing = len(ok) - sum(ok)
    ex = ex.filter(pl.Series(ok)).with_columns(
        synthetic=pl.col("qlang").str.starts_with("syn") | pl.col("tlang").str.starts_with("syn")
    )
    ex.write_parquet(out / "examples.parquet")
    tx.write_parquet(out / "texts.parquet")
    dev.select("ref", "title").write_parquet(out / "dev_d1.parquet")
    stats = {
        "examples": ex.height,
        "by_task": dict(ex.group_by("task").len().sort("task").iter_rows()),
        "by_group": dict(ex.group_by("group").len().sort("group").iter_rows()),
        "dropped_missing_text": missing,
        "dev_d1": {"works": len(held), "pairs": dev.height, "refs_excluded": len(held_refs)},
        "d3a": {
            "translated": n_syn,
            "used": syn.height,
            "en2he_with_negatives": ex.filter(
                (pl.col("task") == "d3a_en2he") & (pl.col("negatives").list.len() > 0)
            ).height,
        },
        "d3b": n_d3b,
        "texts": tx.height,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    log.info("mix: %s", stats)
    return stats
