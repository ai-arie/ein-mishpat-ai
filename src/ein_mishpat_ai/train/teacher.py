"""Teacher pass (configs/teacher.yaml). Layout under data/train/teacher/:

docs_<lang>.parquet        id (ref), text: every text of a training pair (documents, no prompt)
q_<task>_<lang>.parquet    id (ref), text: query texts of a task (encoded with its instruction)
queries.parquet            task, query_ref, positives, title: mining spec, one row per query
pools.parquet              task, ref: target pool of each link task (Hebrew)
jobs.yaml                  encoding jobs for scripts/run_parallel.py
negatives/<task>.parquet   query_ref, cands (positives first, then negatives), n_pos,
                           s_<qlang>_<tlang>: teacher scores of the query vs each candidate
                           (null where a text is missing in that language)

Link pairs are directed "later work → earlier work" (find the source); commentaries by Sefaria's
base_text_titles/dependence first. Only single-segment query sides become queries; target sides
(≤ 5 segments) are the positives. Every ref here comes from data/train/d{1,2} and is therefore
training-eligible (project rules 1-2 (README); tests/test_leakage.py).
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import shlex
from collections import defaultdict
from typing import Any

import numpy as np
import polars as pl
import yaml

from ein_mishpat_ai.bench.kesher import _interleave
from ein_mishpat_ai.utils.io import load_yaml, resolve
from ein_mishpat_ai.utils.log import get_logger

log = get_logger("teacher", "teacher.log")
LANGS = ("he", "en")
Meta = tuple[float | None, bool, frozenset[str]]


def _first_year(v: Any) -> float | None:
    if isinstance(v, list):
        v = v[0] if v else None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def work_meta(schemas_dir: str) -> dict[str, Meta]:
    """title -> (composition year, is a dependent work, base_text_titles)."""
    out: dict[str, Meta] = {}
    for f in glob.glob(f"{schemas_dir}/*.json"):
        try:
            s = json.load(open(f))
        except ValueError:
            continue
        bt = frozenset(b["en"] for b in s.get("base_text_titles") or [] if isinstance(b, dict))
        year = _first_year(s.get("compDate"))
        if year is None:
            year = _first_year(s.get("pubDate"))
        out[s["title"]] = (year, bool(s.get("dependence")), bt)
    return out


def _coin(key: Any, seed: int) -> bool:
    return int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest(), 16) & 1 == 1


def query_is_a(ta: str, tb: str, key: Any, seed: int, meta: dict[str, Meta]) -> bool:
    """Is side a the query (the later / dependent work)?"""
    none: Meta = (None, False, frozenset())
    ya, da, ba = meta.get(ta, none)
    yb, db, bb = meta.get(tb, none)
    if tb in ba and ta not in bb:
        return True
    if ta in bb and tb not in ba:
        return False
    if ta.endswith(f" on {tb}") != tb.endswith(f" on {ta}"):
        return ta.endswith(f" on {tb}")
    if da != db:
        return da
    if ya is not None and yb is not None and ya != yb:
        return ya > yb
    return _coin(key, seed)


def _texts(cfg: dict[str, Any], refs: set[str]) -> pl.DataFrame:
    t = pl.read_parquet(
        resolve(cfg["processed"]["texts"]),
        columns=["ref", "lang", "title", "node_ref", "seg_order", "text_clean", "text_plain"],
    ).filter(pl.col("ref").is_in(list(refs)))
    t = t.with_columns(
        pl.when(pl.col("lang") == "he")
        .then(pl.col("text_plain"))
        .otherwise(pl.col("text_clean"))
        .alias("text")
    ).drop("text_clean", "text_plain")
    return t.filter(pl.col("text").str.strip_chars() != "")


def prepare(cfg: dict[str, Any]) -> dict[str, Any]:
    tc = load_yaml("configs/teacher.yaml")
    out = resolve(tc["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    d1 = pl.read_parquet(resolve("data/train/d1_bitext.parquet"), columns=["ref", "title"])
    d2 = pl.read_parquet(
        resolve("data/train/d2_links.parquet"), columns=["link_id", "a_refs", "b_refs", "task"]
    )
    refs = set(d1["ref"].to_list())
    for c in ("a_refs", "b_refs"):
        refs |= set(d2[c].explode().to_list())
    texts = _texts(cfg, refs)
    has = {lang: set(texts.filter(pl.col("lang") == lang)["ref"].to_list()) for lang in LANGS}
    title_of = dict(texts.select("ref", "title").unique("ref").iter_rows())
    for lang in LANGS:
        texts.filter(pl.col("lang") == lang).select(pl.col("ref").alias("id"), "text").sort(
            "id"
        ).write_parquet(out / f"docs_{lang}.parquet")

    meta = work_meta(str(resolve("data/raw/sefaria/schemas")))
    seed = tc["seed"]
    pos: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    pools: dict[str, set[str]] = defaultdict(set)
    n_dir = defaultdict(int)
    for lid, a, b, task in d2.iter_rows():
        qa = query_is_a(title_of.get(a[0], ""), title_of.get(b[0], ""), lid, seed, meta)
        q, t = (a, b) if qa else (b, a)
        t_he = [x for x in t if x in has["he"]]
        pools[task].update(t_he)
        n_dir["a_query" if qa else "b_query"] += 1
        if len(q) == 1 and q[0] in has["he"] and t_he and q[0] not in t_he:
            pos[task][q[0]].update(t_he)

    rows = []
    stats: dict[str, Any] = {"direction": dict(n_dir), "tasks": {}}
    for task, t in tc["tasks"].items():
        if t["source"] == "d1":
            chosen = d1["ref"].to_list()
            ps = {r: {r} for r in chosen}
        else:
            ps = pos.get(task, {})
            cands = [(r, title_of[r]) for r in ps]
            chosen = _interleave(cands, tc["max_queries_per_task"], seed)
        rows += [(task, r, sorted(ps[r]), title_of.get(r, "")) for r in chosen]
        for lang in (t["qlang"],) if t["source"] == "d1" else LANGS:
            q = texts.filter((pl.col("lang") == lang) & pl.col("ref").is_in(chosen))
            q.select(pl.col("ref").alias("id"), "text").sort("id").write_parquet(
                out / f"q_{task}_{lang}.parquet"
            )
            stats["tasks"].setdefault(task, {})[f"queries_{lang}"] = q.height
        stats["tasks"][task]["candidates"] = len(ps)
        stats["tasks"][task]["pool"] = len(pools.get(task, ())) or len(has[t.get("tlang", "he")])
    pl.DataFrame(
        rows, schema=["task", "query_ref", "positives", "title"], orient="row"
    ).write_parquet(out / "queries.parquet")
    pl.DataFrame(
        [(task, r) for task, rs in pools.items() for r in sorted(rs)],
        schema=["task", "ref"],
        orient="row",
    ).write_parquet(out / "pools.parquet")
    stats["docs"] = {lang: len(has[lang]) for lang in LANGS}
    write_jobs(tc)
    (out / "prepare_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("prepare: %s", stats)
    return stats


def write_jobs(tc: dict[str, Any]) -> None:
    out = resolve(tc["out_dir"])
    model = tc["model"]
    jobs = []
    for f in sorted(out.glob("*.parquet")):
        name = f.stem
        if not (name.startswith("docs_") or name.startswith("q_")):
            continue
        n = pl.read_parquet(f, columns=["id"]).height
        if not n:
            continue
        is_q = name.startswith("q_")
        prefix = ""
        if is_q:
            task = name[2:].rsplit("_", 1)[0]
            prefix = f"Instruct: {tc['tasks'][task]['instruction']}\\nQuery: "
        shards = max(1, math.ceil(n / tc["rows_per_job"]))
        for k in range(shards):
            cmd = (
                f"python scripts/encode_corpus.py {model} {tc['out_dir']}/{name}.parquet "
                f"{tc['emb_dir']}/{model}/{name} --shard {k} --num-shards {shards} "
                f"--max-length {tc['max_length']['query' if is_q else 'doc']} "
                f"--batch-size {tc['batch_size']}"
            )
            if prefix:
                cmd += f" --prefix {shlex.quote(prefix)}"
            jobs.append(({"name": f"{name}-s{k}", "cmd": cmd, "gpus": 1}, n / shards, is_q))
    jobs.sort(key=lambda j: (j[2], -j[1]))  # documents (long texts) first
    (out / "jobs.yaml").write_text(
        yaml.safe_dump({"batch": "m3a_teacher_encode", "jobs": [j[0] for j in jobs]}, width=1000)
    )
    mine_jobs = [
        {"name": f"mine-{t}", "cmd": f"python scripts/teacher.py mine --task {t}", "gpus": 1}
        for t in tc["tasks"]
    ]
    (out / "mine_jobs.yaml").write_text(
        yaml.safe_dump({"batch": "m3a_mine", "jobs": mine_jobs}, width=1000)
    )


def _emb(tc: dict[str, Any], name: str) -> tuple[dict[str, int], np.ndarray]:
    from ein_mishpat_ai.eval.encode import load_embeddings

    ids, m = load_embeddings(resolve(tc["emb_dir"]) / tc["model"] / name)
    n = pl.read_parquet(resolve(tc["out_dir"]) / f"{name}.parquet", columns=["id"]).height
    if len(ids) != n:  # an encoding job failed or is still running
        raise SystemExit(f"{name}: {len(ids)} of {n} rows encoded")
    return {r: i for i, r in enumerate(ids)}, m


def _linked(cfg: dict[str, Any], refs: set[str]) -> dict[str, set[str]]:
    """ref -> refs linked to it anywhere in the link graph (all link types, all splits)."""
    lk = pl.read_parquet(resolve(cfg["processed"]["links"]), columns=["refs1", "refs2"]).filter(
        pl.col("refs1").is_not_null() & pl.col("refs2").is_not_null()
    )
    e = lk.explode("refs1").explode("refs2").rename({"refs1": "a", "refs2": "b"})
    e = pl.concat([e, e.select(pl.col("b").alias("a"), pl.col("a").alias("b"))])
    e = e.filter(pl.col("a").is_in(list(refs)))
    out: dict[str, set[str]] = defaultdict(set)
    for a, b in e.iter_rows():
        out[a].add(b)
    return out


def _scores(qm: np.ndarray, qi: np.ndarray, cm: np.ndarray, ci: np.ndarray) -> np.ndarray:
    """Teacher score of query qi[n] vs candidates ci[n, k] (-1 = missing → nan)."""
    import torch

    out = np.full(ci.shape, np.nan, np.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for s in range(0, len(qi), 2048):
        q_idx, c_idx = qi[s : s + 2048], ci[s : s + 2048]
        ok = (q_idx[:, None] >= 0) & (c_idx >= 0)
        q = torch.from_numpy(qm[np.maximum(q_idx, 0)]).to(dev).float()
        c = torch.from_numpy(cm[np.maximum(c_idx, 0)]).to(dev).float()
        v = torch.einsum("nd,nkd->nk", q, c).cpu().numpy()
        out[s : s + 2048] = np.where(ok, v, np.nan)
    return out


def false_negative_filter(
    cfg: dict[str, Any],
    mc: dict[str, Any],
    queries: list[str],
    positives: list[list[str]],
    top_s: np.ndarray,
    top_i: np.ndarray,
    pool_refs: list[str],
    pos_best: np.ndarray,
) -> tuple[list[list[tuple[str, float]]], dict[str, int]]:
    """Per query, the first `mc["keep"]` retrieved (ref, score) that are not false negatives: not a
    positive or the query, not within ±`neighbours` segments of either, not linked to either, and
    not scoring > `max_rel_score` × the best positive. Also the counts of removed candidates."""
    texts = pl.read_parquet(
        resolve(cfg["processed"]["texts"]), columns=["ref", "node_ref", "seg_order"]
    ).unique("ref")
    where = {r: (n, o) for r, n, o in texts.iter_rows()}
    need = set(queries) | {p for ps in positives for p in ps}
    linked = _linked(cfg, need)

    removed: dict[str, int] = defaultdict(int)
    keeps = []
    for n, (q, ps) in enumerate(zip(queries, positives, strict=True)):
        anchors = [q, *ps]
        near = defaultdict(list)
        for a in anchors:
            if a in where:
                near[where[a][0]].append(where[a][1])
        bad = set(anchors)
        for a in anchors:
            bad |= linked.get(a, set())
        keep = []
        for s, j in zip(top_s[n], top_i[n], strict=True):
            r = pool_refs[j]
            if r in ps or r == q:
                removed["positive"] += 1
            elif r in where and any(
                abs(where[r][1] - o) <= mc["neighbours"] for o in near.get(where[r][0], ())
            ):
                removed["neighbour"] += 1
            elif r in bad:
                removed["linked"] += 1
            elif s > mc["max_rel_score"] * pos_best[n]:
                removed["score"] += 1
            else:
                keep.append((r, float(s)))
                if len(keep) == mc["keep"]:
                    break
        keeps.append(keep)
    return keeps, removed


def mine(cfg: dict[str, Any], task: str, config: str = "configs/teacher.yaml") -> dict[str, Any]:
    from ein_mishpat_ai.eval.metrics import topk

    tc = load_yaml(config)
    mc = tc["mining"]
    t = tc["tasks"][task]
    out = resolve(tc["out_dir"])
    spec = pl.read_parquet(out / "queries.parquet").filter(pl.col("task") == task)
    qlang, tlang = (t["qlang"], t["tlang"]) if t["source"] == "d1" else ("he", "he")
    docs = {lang: _emb(tc, f"docs_{lang}") for lang in LANGS}
    qemb = {}
    for lang in LANGS:
        if (resolve(tc["emb_dir"]) / tc["model"] / f"q_{task}_{lang}").exists():
            qemb[lang] = _emb(tc, f"q_{task}_{lang}")

    d_idx, d_mat = docs[tlang]
    if t["source"] == "d1":
        pool_refs = sorted(d_idx)
    else:
        pool_refs = sorted(
            set(
                pl.read_parquet(out / "pools.parquet")
                .filter(pl.col("task") == task)["ref"]
                .to_list()
            )
            & set(d_idx)
        )
    pool_rows = np.array([d_idx[r] for r in pool_refs])
    pool = d_mat[pool_rows]
    q_idx, q_mat = qemb[qlang]
    spec = spec.filter(pl.col("query_ref").is_in(list(q_idx)))
    queries = spec["query_ref"].to_list()
    positives = spec["positives"].to_list()
    qv = q_mat[[q_idx[r] for r in queries]]
    chunk = max(256, min(4096, int(4e9 // max(1, len(pool_refs)))))
    log.info("%s: %d queries, pool %d (%s→%s)", task, len(queries), len(pool_refs), qlang, tlang)
    top_s, top_i = topk(qv, pool, mc["depth"], chunk=chunk)

    pos_rows = [[d_idx[p] for p in ps if p in d_idx] for ps in positives]
    pos_best = np.array(
        [
            float((d_mat[r].astype(np.float32) @ q.astype(np.float32)).max()) if r else np.nan
            for r, q in zip(pos_rows, qv, strict=True)
        ]
    )
    keeps, removed = false_negative_filter(
        cfg, mc, queries, positives, top_s, top_i, pool_refs, pos_best
    )
    cands = [[*ps, *[r for r, _ in kp]] for ps, kp in zip(positives, keeps, strict=True)]
    n_pos = [len(ps) for ps in positives]

    width = max(len(c) for c in cands)
    res = {"query_ref": queries, "cands": cands, "n_pos": n_pos}
    for ql in LANGS:
        for tl in LANGS:
            if ql not in qemb or (t["source"] == "d1" and (ql, tl) != (qlang, tlang)):
                continue
            qi_map, qm = qemb[ql]
            ci_map, cm = docs[tl]
            qi = np.array([qi_map.get(r, -1) for r in queries])
            ci = np.full((len(cands), width), -1, np.int64)
            for n, c in enumerate(cands):
                ci[n, : len(c)] = [ci_map.get(r, -1) for r in c]
            sc = _scores(qm, qi, cm, ci)
            res[f"s_{ql}_{tl}"] = [
                [None if math.isnan(x) else float(x) for x in sc[n, : len(c)]]
                for n, c in enumerate(cands)
            ]
    nd = resolve(tc["out_dir"]) / "negatives"
    nd.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(res)
    df = df.with_columns(pl.col(c).cast(pl.List(pl.Float32)) for c in df.columns if c[:2] == "s_")
    df.write_parquet(nd / f"{task}.parquet")
    n_neg = np.array([len(c) - p for c, p in zip(cands, n_pos, strict=True)])
    stats = {
        "task": task,
        "queries": len(queries),
        "pool": len(pool_refs),
        "removed": dict(removed),
        "negatives_mean": float(n_neg.mean()) if len(n_neg) else 0.0,
        "queries_lt_keep": int((n_neg < mc["keep"]).sum()),
        "pos_best_mean": float(np.nanmean(pos_best)) if len(pos_best) else None,
        "top1_is_positive": float(
            np.mean([pool_refs[top_i[n][0]] in ps for n, ps in enumerate(positives)])
        )
        if positives
        else None,
    }
    (nd / f"{task}.stats.json").write_text(json.dumps(stats, indent=2))
    log.info("mined %s: %s", task, stats)
    return stats
