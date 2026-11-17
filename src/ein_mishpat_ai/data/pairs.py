"""Text table and training pairs.

data/processed/texts.parquet  one row per (ref, lang): the preferred version's text, with `split`
                              (train/dev/test) and `leak` (Sefaria-bench ref ±2). Shared by
                              KesherBench and training.
data/train/d1_bitext.parquet  (ref) pairs with both Hebrew/Aramaic and English, training works only
data/train/d2_links.parquet   (a_refs, b_refs, task) link pairs, both sides training-eligible

Pairs store refs only; loaders join texts.parquet. A pair is training-eligible only if every ref on
both sides has split == "train", leak == false, and is not a KesherBench dev/test query
(project rules 1-2 (README)). Build KesherBench before the pairs.
"""

from __future__ import annotations

import json
from typing import Any

import polars as pl

from ein_mishpat_ai.data.leakage import excluded_refs
from ein_mishpat_ai.data.splits import Splits
from ein_mishpat_ai.utils.io import resolve
from ein_mishpat_ai.utils.log import get_logger

log = get_logger("pairs", "build_data.log")

LICENSE_RANK = {"public_domain": 0, "cc0": 1, "cc_by": 2, "cc_by_sa": 3, "unknown": 4}
TASK = {
    "ein mishpat / ner mitsvah": "ein_mishpat",
    "ein mishpat": "ein_mishpat",
    "commentary": "commentary",
    "super_commentary": "commentary",
    "super commentary": "commentary",
    "parshanut": "commentary",
    "dibur_hamatchil": "commentary",
    "mesorat hashas": "mesorat_hashas",
    "related": "parallel",
    "related passage": "parallel",
    "mishnah in talmud": "mishnah_gemara",
    "quotation": "quotation",
    "quotation_auto": "quotation",
    "quotation_auto_tanakh": "quotation",
    "midrash": "midrash",
    "targum": "targum",
    "sifrei mitzvot": "sifrei_mitzvot",
    "(empty)": "untyped",
}


def build_texts(cfg: dict[str, Any]) -> pl.DataFrame:
    p = cfg["processed"]
    versions = pl.read_parquet(resolve(p["versions"]))
    seg = pl.scan_parquet(resolve(p["segments"]))
    size = seg.group_by("version_id").agg(pl.len().alias("_size"))
    ranked = (
        seg.join(size, on="version_id")
        .join(versions.lazy().select("version_id", "version_title"), on="version_id")
        .with_columns(pl.col("license").replace_strict(LICENSE_RANK).alias("_lic"))
        .sort(
            ["ref", "lang", "synthetic", "_lic", "_size", "version_title"],
            descending=[False, False, False, False, True, False],
        )
        .unique(["ref", "lang"], keep="first", maintain_order=True)
        .drop("_lic", "_size", "text_raw")
        .collect()
    )
    sp = Splits()
    keys = ranked.select("title", "node_ref").unique()
    keys = keys.with_columns(
        pl.struct("title", "node_ref")
        .map_elements(
            lambda r: sp.split_of(r["title"], r["node_ref"]) or "train", return_dtype=pl.Utf8
        )
        .alias("split")
    )
    leak = excluded_refs(cfg)
    texts = ranked.join(keys, on=["title", "node_ref"], how="left").with_columns(
        pl.col("ref").is_in(list(leak)).alias("leak")
    )
    out = resolve(p["texts"])
    out.parent.mkdir(parents=True, exist_ok=True)
    texts.write_parquet(out, compression="zstd", row_group_size=200_000)
    log.info("texts: %d rows (%s)", texts.height, dict(texts.group_by("split").len().iter_rows()))
    return texts


def build_pairs(cfg: dict[str, Any]) -> dict[str, Any]:
    p = cfg["processed"]
    texts = pl.read_parquet(
        resolve(p["texts"]),
        columns=["ref", "lang", "split", "leak", "synthetic", "category", "title"],
    )
    ok_refs = texts.filter((pl.col("split") == "train") & ~pl.col("leak")).select("ref").unique()
    he = texts.filter(pl.col("lang") == "he").select("ref", pl.col("synthetic").alias("he_syn"))
    en = texts.filter(pl.col("lang") == "en").select("ref", pl.col("synthetic").alias("en_syn"))
    allowed = set(ok_refs["ref"].to_list())
    bad = set(texts.filter((pl.col("split") != "train") | pl.col("leak"))["ref"].to_list())
    allowed -= bad
    # KesherBench queries (dev/test, any task) never appear in a training pair: a dev query can
    # come from a training work and also have positives in training works (rule 1 test).
    eval_q: set[str] = set()
    for f in resolve("data/bench/kesher").glob("*/qrels_*.parquet"):
        eval_q |= set(pl.read_parquet(f, columns=["query_id"])["query_id"].to_list())
    allowed -= eval_q

    d1 = (
        he.join(en, on="ref")
        .filter(pl.col("ref").is_in(list(allowed)))
        .join(texts.filter(pl.col("lang") == "he").select("ref", "title", "category"), on="ref")
        .with_columns(
            (pl.col("he_syn") | pl.col("en_syn")).alias("synthetic"), pl.lit("bitext").alias("task")
        )
        .select("ref", "title", "category", "task", "synthetic")
    )
    out_dir = resolve("data/train")
    out_dir.mkdir(parents=True, exist_ok=True)
    d1.write_parquet(out_dir / "d1_bitext.parquet")

    links = pl.read_parquet(
        resolve(p["links"]),
        columns=[
            "link_id",
            "conn_type",
            "status1",
            "status2",
            "refs1",
            "refs2",
            "title1",
            "title2",
            "cat1",
            "cat2",
        ],
    )
    links = links.filter((pl.col("status1") == "ok") & (pl.col("status2") == "ok"))
    has_he = set(he["ref"].to_list())
    has_en = set(en["ref"].to_list())
    rows = []
    dropped = {"heldout_or_leak": 0, "same_refs": 0}
    for r in links.iter_rows(named=True):
        a, b = r["refs1"], r["refs2"]
        if a == b:
            dropped["same_refs"] += 1
            continue
        if not all(x in allowed for x in a) or not all(x in allowed for x in b):
            dropped["heldout_or_leak"] += 1
            continue
        rows.append(
            (
                r["link_id"],
                a,
                b,
                r["title1"],
                r["title2"],
                r["cat1"],
                r["cat2"],
                r["conn_type"],
                TASK.get(r["conn_type"], "other"),
                all(x in has_he for x in a),
                all(x in has_he for x in b),
                all(x in has_en for x in a),
                all(x in has_en for x in b),
            )
        )
    d2 = pl.DataFrame(
        rows,
        schema=[
            "link_id",
            "a_refs",
            "b_refs",
            "a_title",
            "b_title",
            "a_cat",
            "b_cat",
            "conn_type",
            "task",
            "a_he",
            "b_he",
            "a_en",
            "b_en",
        ],
        orient="row",
    )
    d2 = (
        d2.with_columns(
            pl.concat_str(
                pl.col("a_refs").list.join("|"), pl.col("b_refs").list.join("|"), separator="#"
            ).alias("_k1"),
            pl.concat_str(
                pl.col("b_refs").list.join("|"), pl.col("a_refs").list.join("|"), separator="#"
            ).alias("_k2"),
        )
        .with_columns(pl.min_horizontal("_k1", "_k2").alias("_k"))
        .unique(["_k", "task"], keep="first", maintain_order=True)
        .drop("_k1", "_k2", "_k")
    )
    d2.write_parquet(out_dir / "d2_links.parquet")
    stats = {
        "d1_bitext": d1.height,
        "d1_synthetic": int(d1["synthetic"].sum()),
        "d2_links": d2.height,
        "d2_dropped": dropped,
        "kesher_query_refs_excluded": len(eval_q),
        "d2_by_task": dict(d2.group_by("task").len().sort("len", descending=True).iter_rows()),
        "d2_he_he": int((d2["a_he"] & d2["b_he"]).sum()),
        "d2_with_en_side": int((d2["a_en"] | d2["b_en"]).sum()),
    }
    (out_dir / "pairs_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("pairs: %s", stats)
    return stats
