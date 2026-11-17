"""M0 pilot task: queries = Mishneh Torah halachot (he; en where a kept English version exists),
corpus = all Bavli segments (Wikisource, he), positives = Ein Mishpat links (segment level)."""

from __future__ import annotations

from typing import Any

import polars as pl

from ein_mishpat_ai.utils.io import load_yaml, resolve
from ein_mishpat_ai.utils.log import get_logger

log = get_logger("pilot", "bench.log")


def build(cfg_path: str = "configs/eval/pilot_mt_bavli.yaml") -> dict[str, Any]:
    cfg = load_yaml(cfg_path)
    data = load_yaml("configs/data.yaml")["processed"]
    versions = pl.read_parquet(resolve(data["versions"]))
    seg = pl.scan_parquet(resolve(data["segments"]))

    corpus = (
        seg.filter(pl.col("categories").str.starts_with("Talmud > Bavli > Seder"))
        .join(versions.lazy().select("version_id", "version_title"), on="version_id")
        .filter((pl.col("version_title") == cfg["corpus_version"]) & ~pl.col("synthetic"))
        .select(pl.col("ref").alias("id"), pl.col("text_clean").alias("text"), "title")
        .collect()
    )
    corpus_ids = set(corpus["id"].to_list())

    links = pl.read_parquet(resolve(data["links"])).filter(
        pl.col("conn_type").is_in(cfg["conn_types"])
    )
    pairs = []
    for row in links.iter_rows(named=True):
        for a, b in (("1", "2"), ("2", "1")):
            if (
                row[f"title{a}"].startswith("Mishneh Torah, ")
                and row[f"status{a}"] == "ok"
                and row[f"n{a}"] == 1
                and row[f"status{b}"] == "ok"
            ):
                for t in row[f"refs{b}"]:
                    if t in corpus_ids:
                        pairs.append((row[f"refs{a}"][0], t))
    qrels_all = pl.DataFrame(pairs, schema=["query_id", "corpus_id"], orient="row").unique()

    mt = (
        seg.filter(pl.col("title").str.starts_with("Mishneh Torah, "))
        .join(versions.lazy().select("version_id", "version_title"), on="version_id")
        .filter(~pl.col("synthetic"))  # rule 4: no machine/LLM text in evaluation
        .select("ref", "lang", "version_title", "text_clean", "license")
        .collect()
    )
    rank = {v: i for i, v in enumerate(cfg["query_he_versions"])}
    he = (
        mt.filter(
            (pl.col("lang") == "he") & pl.col("version_title").is_in(cfg["query_he_versions"])
        )
        .with_columns(pl.col("version_title").replace_strict(rank).alias("_r"))
        .sort("ref", "_r")
        .unique("ref", keep="first")
    )
    en = mt.filter(pl.col("lang") == "en").sort("ref", "version_title").unique("ref", keep="first")
    candidates = sorted(set(qrels_all["query_id"].to_list()) & set(he["ref"].to_list()))
    sample = (
        pl.DataFrame({"query_id": candidates})
        .sample(n=min(cfg["n_queries"], len(candidates)), seed=cfg["seed"])
        .sort("query_id")
    )
    qrels = qrels_all.join(sample, on="query_id")
    q_he = he.join(sample, left_on="ref", right_on="query_id").select(
        pl.col("ref").alias("id"), pl.col("text_clean").alias("text"), "version_title"
    )
    q_en = en.join(sample, left_on="ref", right_on="query_id").select(
        pl.col("ref").alias("id"), pl.col("text_clean").alias("text"), "version_title"
    )
    out = resolve(cfg["task_dir"])
    out.mkdir(parents=True, exist_ok=True)
    corpus.select("id", "text").write_parquet(out / "corpus.parquet")
    q_he.write_parquet(out / "queries_he.parquet")
    q_en.write_parquet(out / "queries_en.parquet")
    qrels.write_parquet(out / "qrels.parquet")
    stats = {
        "corpus": corpus.height,
        "candidate_queries": len(candidates),
        "queries_he": q_he.height,
        "queries_en": q_en.height,
        "qrels": qrels.height,
        "positives_per_query": round(qrels.height / max(1, sample.height), 2),
        "query_versions_he": dict(q_he["version_title"].value_counts().iter_rows()),
        "query_versions_en": dict(q_en["version_title"].value_counts().iter_rows()),
    }
    log.info("pilot task: %s", stats)
    return stats
