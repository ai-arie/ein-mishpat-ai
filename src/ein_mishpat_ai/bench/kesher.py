"""KesherBench construction (configs/kesher.yaml). Layout under data/bench/kesher/:

corpora/<name>_<lang>.parquet              id (ref), text, title, license
<task>/queries_<setting>_<split>.parquet   id, text, title, license, strict
<task>/qrels_<split>.parquet               query_id, corpus_id, conn_type
<task>/task.json                           corpus, settings, counts
"""

from __future__ import annotations

import glob
import json
import random
from collections import defaultdict
from typing import Any

import polars as pl

from ein_mishpat_ai.data.normalize import clean
from ein_mishpat_ai.utils.io import load_yaml, resolve
from ein_mishpat_ai.utils.log import get_logger

log = get_logger("kesher", "bench.log")


def _texts(cfg: dict[str, Any]) -> pl.DataFrame:
    t = pl.read_parquet(
        resolve(cfg["processed"]["texts"]),
        columns=[
            "ref",
            "lang",
            "title",
            "categories",
            "license",
            "synthetic",
            "split",
            "text_clean",
            "text_plain",
        ],
    )
    t = t.filter(~pl.col("synthetic"))  # rule 4
    return t.with_columns(
        pl.when(pl.col("lang") == "he")
        .then(pl.col("text_plain"))
        .otherwise(pl.col("text_clean"))
        .alias("text")
    ).drop("text_clean", "text_plain", "synthetic")


def _interleave(ids_titles: list[tuple[str, str]], cap: int, seed: int) -> list[str]:
    """Up to `cap` ids, round-robin over query works (titles), shuffled within each work."""
    rng = random.Random(seed)
    by: dict[str, list[str]] = defaultdict(list)
    for i, t in ids_titles:
        by[t].append(i)
    groups = sorted(by)
    rng.shuffle(groups)
    for g in groups:
        rng.shuffle(by[g])
    out: list[str] = []
    while len(out) < cap and any(by[g] for g in groups):
        for g in groups:
            if by[g] and len(out) < cap:
                out.append(by[g].pop())
    return sorted(out)


def _query_mask(df: pl.DataFrame, q: dict[str, Any], corpus_prefix: str) -> pl.Series:
    m = pl.Series([True] * df.height)
    if "category_prefix" in q:
        m &= df["categories"].str.starts_with(q["category_prefix"])
    if "title_prefix" in q:
        m &= df["title"].str.starts_with(q["title_prefix"])
    if "title_regex" in q:
        m &= df["title"].str.contains(q["title_regex"])
    if q.get("exclude_corpus"):
        m &= ~df["categories"].str.starts_with(corpus_prefix)
    return m


def build_corpora(kcfg: dict[str, Any], texts: pl.DataFrame) -> dict[str, set[str]]:
    out = resolve(kcfg["out_dir"]) / "corpora"
    out.mkdir(parents=True, exist_ok=True)
    ids: dict[str, set[str]] = {}
    for name, c in kcfg["corpora"].items():
        for lang in c["langs"]:
            df = texts.filter(
                (pl.col("lang") == lang)
                & pl.col("categories").str.starts_with(c["category_prefix"])
            )
            df.select(pl.col("ref").alias("id"), "text", "title", "license").sort(
                "id"
            ).write_parquet(out / f"{name}_{lang}.parquet")
            ids[f"{name}_{lang}"] = set(df["ref"].to_list())
            log.info("corpus %s_%s: %d", name, lang, df.height)
    return ids


def build_link_task(
    name: str,
    t: dict[str, Any],
    kcfg: dict[str, Any],
    cfg: dict[str, Any],
    texts: pl.DataFrame,
    corpus_ids: dict[str, set[str]],
) -> dict[str, Any]:
    corpus = t["corpus"]
    cprefix = kcfg["corpora"][corpus]["category_prefix"]
    he_corpus = corpus_ids[f"{corpus}_he"]
    he = texts.filter(pl.col("lang") == "he")
    qcands = he.filter(_query_mask(he, t["query"], cprefix))
    q_title = dict(zip(qcands["ref"], qcands["title"], strict=True))
    split_of = dict(zip(he["ref"], he["split"], strict=True))

    links = pl.read_parquet(
        resolve(cfg["processed"]["links"]),
        columns=["conn_type", "status1", "status2", "refs1", "refs2", "n1", "n2"],
    )
    links = links.filter(
        pl.col("conn_type").is_in(t["conn_types"])
        & (pl.col("status1") == "ok")
        & (pl.col("status2") == "ok")
    )
    pos: dict[str, dict[str, str]] = defaultdict(dict)
    for c, r1, r2, n1, n2 in links.select("conn_type", "refs1", "refs2", "n1", "n2").iter_rows():
        for qa, ca, nq, nc in ((r1, r2, n1, n2), (r2, r1, n2, n1)):
            if nq > kcfg["max_query_segments"] or nc > kcfg["max_positive_segments"]:
                continue
            q = qa[0]
            if q in q_title and all(x in he_corpus for x in ca) and q not in ca:
                for x in ca:
                    pos[q].setdefault(x, c)

    by_split: dict[str, list[tuple[str, str]]] = {"test": [], "dev": []}
    for q, ps in pos.items():
        sps = {split_of.get(x, "train") for x in ps}
        s = "test" if "test" in sps else "dev" if "dev" in sps else None
        if s:
            by_split[s].append((q, q_title[q]))

    out = resolve(kcfg["out_dir"]) / name
    out.mkdir(parents=True, exist_ok=True)
    en = texts.filter(pl.col("lang") == "en")
    counts: dict[str, Any] = {}
    for s, cands in by_split.items():
        chosen = _interleave(cands, kcfg["caps"][s], kcfg["seed"])
        qrels = pl.DataFrame(
            [(q, x, c) for q in chosen for x, c in sorted(pos[q].items())],
            schema=["query_id", "corpus_id", "conn_type"],
            orient="row",
        )
        qrels.write_parquet(out / f"qrels_{s}.parquet")
        counts[f"{s}/candidates"] = len(cands)
        counts[f"{s}/qrels"] = qrels.height
        for setting in t["settings"]:
            qlang, clang = setting[:2], setting[3:]
            src = he if qlang == "he" else en
            q = src.filter(pl.col("ref").is_in(chosen)).select(
                pl.col("ref").alias("id"),
                "text",
                "title",
                "license",
                (pl.col("split") == s).alias("strict"),
            )
            if clang == "en":
                cen = corpus_ids.get(f"{corpus}_en", set())
                keep = {r for r in qrels["query_id"].unique() if any(x in cen for x in pos[r])}
                q = q.filter(pl.col("id").is_in(list(keep)))
            qfile = out / f"queries_{setting}_{s}.parquet"
            if q.height:
                q.sort("id").write_parquet(qfile)
            elif qfile.exists():  # stale file from an earlier build
                qfile.unlink()
            counts[f"{s}/{setting}"] = q.height
            counts[f"{s}/{setting}/strict"] = int(q["strict"].sum()) if q.height else 0
    meta = {
        "task": name,
        "corpus": corpus,
        "settings": t["settings"],
        "conn_types": t["conn_types"],
        "counts": counts,
    }
    (out / "task.json").write_text(json.dumps(meta, indent=2))
    log.info("%s: %s", name, counts)
    return meta


def build_topics(kcfg: dict[str, Any], texts: pl.DataFrame) -> dict[str, Any]:
    tc = kcfg["topics"]
    he = texts.filter(pl.col("lang") == "he")
    he_ids = set(he["ref"].to_list())
    rng = random.Random(kcfg["seed"])
    topics = []
    for f in sorted(glob.glob(str(resolve("data/raw/sefaria/topics/by_slug/*.json")))):
        x = json.load(open(f))
        srcs: list[list[str]] = []
        for v in (x.get("refs") or {}).values():
            for r in v.get("refs", []) if isinstance(v, dict) else []:
                if not isinstance(r, dict) or r.get("is_sheet"):
                    continue
                if r.get("generatedBy") in tc["exclude_generated_by"]:
                    continue
                ds = set((r.get("dataSources") or {}).keys())
                if ds and ds <= set(tc["user_only_sources"]):
                    continue
                exp = r.get("expandedRefs") or [r.get("ref")]
                if 0 < len(exp) <= kcfg["max_positive_segments"] and all(e in he_ids for e in exp):
                    srcs.append(exp)
        if len(srcs) < tc["min_sources"]:
            continue
        desc = x.get("description") if x.get("description_published") else None
        desc = desc if isinstance(desc, dict) else {}
        title = x.get("primaryTitle") or {}
        q = {}
        for lang in ("en", "he"):
            tt = title.get(lang) or x.get(lang) or ""
            d = clean(desc.get(lang) or "")
            q[lang] = f"{tt}. {d}" if d else tt
        topics.append({"slug": x["slug"], "q": q, "pos": sorted({e for s in srcs for e in s})})
    rng.shuffle(topics)
    n_dev = int(len(topics) * tc["dev_fraction"])
    split = {"dev": topics[:n_dev], "test": topics[n_dev:]}
    positives = {e for t in topics for e in t["pos"]}
    cats = (
        he.filter(pl.col("ref").is_in(list(positives)))["categories"]
        .str.split(" > ")
        .list.first()
        .unique()
        .to_list()
    )
    pool = he.filter(
        pl.col("categories").str.split(" > ").list.first().is_in(cats)
        & ~pl.col("ref").is_in(list(positives))
    )
    pool = pool.sample(n=min(tc["distractors"], pool.height), seed=kcfg["seed"])
    corpus = pl.concat([he.filter(pl.col("ref").is_in(list(positives))), pool])
    out = resolve(kcfg["out_dir"])
    corpus.select(pl.col("ref").alias("id"), "text", "title", "license").sort("id").write_parquet(
        out / "corpora" / "k5pool_he.parquet"
    )
    tdir = out / tc["name"]
    tdir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, Any] = {
        "topics_too_broad": sum(len(t["pos"]) > tc["max_positive_segments"] for t in topics)
    }
    # queries only for topics with a focused source list; the pool (above) is unchanged
    split = {
        s: [t for t in ts if len(t["pos"]) <= tc["max_positive_segments"]]
        for s, ts in split.items()
    }
    for s, ts in split.items():
        ts = ts[: kcfg["caps"][s]]
        pl.DataFrame(
            [(t["slug"], e, "topic") for t in ts for e in t["pos"]],
            schema=["query_id", "corpus_id", "conn_type"],
            orient="row",
        ).write_parquet(tdir / f"qrels_{s}.parquet")
        for setting in tc["settings"]:
            lang = setting[:2]
            q = pl.DataFrame(
                [(t["slug"], t["q"][lang], t["slug"], "", True) for t in ts if t["q"][lang]],
                schema=["id", "text", "title", "license", "strict"],
                orient="row",
            )
            q.write_parquet(tdir / f"queries_{setting}_{s}.parquet")
            counts[f"{s}/{setting}"] = q.height
        counts[f"{s}/qrels"] = sum(len(t["pos"]) for t in ts)
    counts["topics_eligible"] = len(topics)
    counts["corpus"] = corpus.height
    counts["with_published_description"] = sum(". " in t["q"]["en"] for t in topics)
    meta = {
        "task": tc["name"],
        "corpus": "k5pool",
        "settings": tc["settings"],
        "conn_types": ["topic"],
        "counts": counts,
    }
    (tdir / "task.json").write_text(json.dumps(meta, indent=2))
    log.info("%s: %s", tc["name"], counts)
    return meta


def build_all(cfg: dict[str, Any]) -> dict[str, Any]:
    kcfg = load_yaml("configs/kesher.yaml")
    texts = _texts(cfg)
    corpus_ids = build_corpora(kcfg, texts)
    out = {
        name: build_link_task(name, t, kcfg, cfg, texts, corpus_ids)["counts"]
        for name, t in kcfg["tasks"].items()
    }
    out[kcfg["topics"]["name"]] = build_topics(kcfg, texts)["counts"]
    return out
