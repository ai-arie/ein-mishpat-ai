"""M3 teacher filters. They only add flags; nothing is dropped here
(M4 picks the thresholds).

D3a round trip: each synthetic English translation is a query ("retrieve its translation")
against every training Hebrew document; it passes if the teacher ranks its own source in the top 10.
D1 alignment: teacher score and top-10 rank of the paired side for every D1 bitext pair in both
directions (misaligned pairs, e.g. English that belongs to another passage, rank low).

usage: scripts/filters.py {d3a-prepare,d3a-score,d1-score}; d3a-prepare writes the encoding jobs
(data/gen/d3a/roundtrip/jobs.yaml, for run_parallel.py), re-run it as more D3a blocks finish.
"""

from __future__ import annotations

import json
import shlex
from typing import Any

import numpy as np
import polars as pl
import yaml

from ein_mishpat_ai.train.teacher import _emb
from ein_mishpat_ai.utils.io import load_yaml, resolve
from ein_mishpat_ai.utils.log import get_logger

log = get_logger("filters", "teacher.log")
TOP = 10
SHARDS_PER_BLOCK = 4


def _cfgs() -> tuple[dict[str, Any], dict[str, Any], Any]:
    tc = load_yaml("configs/teacher.yaml")
    dc = load_yaml("configs/gen.yaml")["d3a"]
    return tc, dc, resolve(dc["out_dir"]) / "roundtrip"


def _rank(
    q: np.ndarray, pool: np.ndarray, pool_refs: list[str], own: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Rank (1-based, 0 = not in top 10) of each query's own ref, its score, and the top-1 ref."""
    from ein_mishpat_ai.eval.metrics import topk

    _, top_i = topk(q, pool, TOP, chunk=max(256, min(4096, int(4e9 // len(pool_refs)))))
    rank = np.zeros(len(own), np.int16)
    for n, r in enumerate(own):
        hit = [k for k, j in enumerate(top_i[n]) if pool_refs[j] == r]
        rank[n] = hit[0] + 1 if hit else 0
    return rank, top_i, [pool_refs[j] for j in top_i[:, 0]]


def d3a_prepare() -> dict[str, Any]:
    tc, dc, rt = _cfgs()
    rt.mkdir(parents=True, exist_ok=True)
    d3a = resolve(dc["out_dir"])
    # re-translated batches (gen/d3a.py redo) are scored as their own tables: d3a_en_redo_blockNNN
    blocks = [(f"d3a_en_{f.stem}", f) for f in sorted((d3a / "blocks").glob("block*.parquet"))] + [
        (f"d3a_en_redo_{f.stem}", f) for f in sorted((d3a / "redo").glob("block*.parquet"))
    ]
    prefix = f"Instruct: {tc['tasks']['bitext_en2he']['instruction']}\\nQuery: "
    jobs = []
    for name, f in blocks:
        q = rt / f"{name}.parquet"
        if not q.exists():
            pl.read_parquet(f).filter(pl.col("translation").is_not_null()).select(
                pl.col("ref").alias("id"), pl.col("translation").alias("text")
            ).write_parquet(q)
        for k in range(SHARDS_PER_BLOCK):
            cmd = (
                f"python scripts/encode_corpus.py {tc['model']} {dc['out_dir']}/roundtrip/{q.name} "
                f"{tc['emb_dir']}/{tc['model']}/{name} --shard {k} --num-shards {SHARDS_PER_BLOCK} "
                f"--max-length {tc['max_length']['query']} --batch-size {tc['batch_size']} "
                f"--prefix {shlex.quote(prefix)}"
            )
            jobs.append({"name": f"{name}-s{k}", "cmd": cmd, "gpus": 1})
    (rt / "jobs.yaml").write_text(
        yaml.safe_dump({"batch": "d3a_roundtrip_encode", "jobs": jobs}, width=1000)
    )
    return {"blocks": len(blocks), "jobs": len(jobs)}


def d3a_score() -> dict[str, Any]:
    """Score every encoded block that has no result yet (block files are only written whole)."""
    tc, _, rt = _cfgs()
    d_idx, d_mat = _emb(tc, "docs_he")
    pool_refs = [""] * len(d_idx)
    for r, i in d_idx.items():
        pool_refs[i] = r
    out = {}
    for q in sorted(rt.glob("d3a_en_*block*.parquet")):
        res = rt / f"score_{q.stem.removeprefix('d3a_en_')}.parquet"
        if res.exists():
            continue
        tc_q = {**tc, "out_dir": str(rt)}
        q_idx, q_mat = _emb(tc_q, q.stem)
        refs = sorted(q_idx, key=q_idx.get)
        rank, _, top1 = _rank(q_mat, d_mat, pool_refs, refs)
        s_src = np.einsum(
            "nd,nd->n", q_mat.astype(np.float32), d_mat[[d_idx[r] for r in refs]].astype(np.float32)
        )
        pl.DataFrame(
            {"ref": refs, "rank": rank, "s_src": s_src, "top1_ref": top1, "pass": rank > 0}
        ).write_parquet(res)
        out[q.stem] = {
            "n": len(refs),
            "pass": float((rank > 0).mean()),
            "r1": float((rank == 1).mean()),
        }
        log.info("d3a round trip %s: %s", q.stem, out[q.stem])
    return out


def d3a_mine(docs: tuple[dict[str, int], np.ndarray] | None = None) -> dict[str, Any]:
    """Hard negatives for D3a (synthetic English → Hebrew source) from the round-trip encodings:
    teacher top-`mining.depth` over every training Hebrew document, false negatives removed as in
    the M3a mining (train/teacher.py), with teacher scores for distillation →
    roundtrip/neg_<block>.parquet: ref, negatives, t_pos [score with the source], t_neg.
    Every block (and redo file) without a result yet; `docs` replaces docs_he (tests)."""
    from ein_mishpat_ai.data.download_sefaria import load_cfg
    from ein_mishpat_ai.eval.metrics import topk
    from ein_mishpat_ai.train.teacher import false_negative_filter

    tc, _, rt = _cfgs()
    mc = tc["mining"]
    d_idx, d_mat = docs or _emb(tc, "docs_he")
    pool_refs = [""] * len(d_idx)
    for r, i in d_idx.items():
        pool_refs[i] = r
    todo = [
        q
        for q in sorted(rt.glob("d3a_en_*block*.parquet"))
        if not (rt / f"neg_{q.stem.removeprefix('d3a_en_')}.parquet").exists()
        and (resolve(tc["emb_dir"]) / tc["model"] / q.stem).exists()
    ]
    if not todo:
        return {}
    tc_q = {**tc, "out_dir": str(rt)}
    parts = []
    for q in todo:
        q_idx, q_mat = _emb(tc_q, q.stem)
        refs = [r for r in sorted(q_idx, key=q_idx.get) if r in d_idx]
        qv = q_mat[[q_idx[r] for r in refs]]
        top_s, top_i = topk(qv, d_mat, mc["depth"], chunk=max(256, int(4e9 // len(pool_refs))))
        s_src = np.einsum(
            "nd,nd->n", qv.astype(np.float32), d_mat[[d_idx[r] for r in refs]].astype(np.float32)
        )
        parts.append((q, refs, top_s, top_i, s_src))
    refs = [r for p in parts for r in p[1]]
    keeps, removed = false_negative_filter(
        load_cfg(),
        mc,
        refs,
        [[r] for r in refs],
        np.vstack([p[2] for p in parts]),
        np.vstack([p[3] for p in parts]),
        pool_refs,
        np.concatenate([p[4] for p in parts]),
    )
    out, n = {}, 0
    for q, rs, _, _, s_src in parts:
        kp = keeps[n : n + len(rs)]
        n += len(rs)
        name = q.stem.removeprefix("d3a_en_")
        pl.DataFrame(
            {
                "ref": rs,
                "negatives": [[r for r, _ in k] for k in kp],
                "t_pos": [[float(s)] for s in s_src],
                "t_neg": [[s for _, s in k] for k in kp],
            },
            schema={
                "ref": pl.String,
                "negatives": pl.List(pl.String),
                "t_pos": pl.List(pl.Float32),
                "t_neg": pl.List(pl.Float32),
            },
        ).write_parquet(rt / f"neg_{name}.parquet")
        n_neg = np.array([len(k) for k in kp])
        out[name] = {"queries": len(rs), "negatives_mean": float(n_neg.mean())}
        out[name]["no_negative"] = float((n_neg == 0).mean())
    out["removed"] = dict(removed)
    log.info("d3a mining: %s", out)
    return out


def d1_score() -> dict[str, Any]:
    tc = load_yaml("configs/teacher.yaml")
    docs = {lang: _emb(tc, f"docs_{lang}") for lang in ("he", "en")}
    res: dict[str, Any] = {}
    stats = {}
    for task, ql, tl in (("bitext_he2en", "he", "en"), ("bitext_en2he", "en", "he")):
        q_idx, q_mat = _emb(tc, f"q_{task}_{ql}")
        d_idx, d_mat = docs[tl]
        refs = sorted(q_idx, key=q_idx.get)
        pool_refs = [""] * len(d_idx)
        for r, i in d_idx.items():
            pool_refs[i] = r
        rank, _, _ = _rank(q_mat, d_mat, pool_refs, refs)
        s = np.einsum(
            "nd,nd->n", q_mat.astype(np.float32), d_mat[[d_idx[r] for r in refs]].astype(np.float32)
        )
        res[task] = pl.DataFrame({"ref": refs, f"rank_{ql}2{tl}": rank, f"s_{ql}2{tl}": s})
        stats[task] = {
            "n": len(refs),
            "top10": float((rank > 0).mean()),
            "r1": float((rank == 1).mean()),
        }
    df = res["bitext_he2en"].join(res["bitext_en2he"], on="ref", how="full", coalesce=True)
    df.write_parquet(resolve(tc["out_dir"]) / "d1_align.parquet")
    log.info("d1 alignment: %s", stats)
    (resolve(tc["out_dir"]) / "d1_align.stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def d1_keep() -> dict[str, Any]:
    """Column `keep` in d1_align.parquet per configs/teacher.yaml `d1_filter`."""
    tc = load_yaml("configs/teacher.yaml")
    f = resolve(tc["out_dir"]) / "d1_align.parquet"
    title = (
        pl.read_parquet(resolve(tc["out_dir"]) / "queries.parquet")
        .filter(pl.col("task") == "bitext_he2en")
        .select(pl.col("query_ref").alias("ref"), "title")
    )
    d = pl.read_parquet(f).drop("keep", "title", strict=False).join(title, on="ref", how="left")
    found = (pl.col("rank_he2en") > 0) | (pl.col("rank_en2he") > 0)
    d = d.with_columns(keep=found & ~pl.col("title").is_in(tc["d1_filter"]["drop_works"]))
    d.drop("title").write_parquet(f)
    stats = {"pairs": d.height, "kept": int(d["keep"].sum()), "kept_share": float(d["keep"].mean())}
    log.info("d1 filter: %s", stats)
    return stats


def d3a_keep() -> dict[str, Any]:
    """Column `keep` in every D3a round-trip score file per configs/teacher.yaml `d3a_filter`."""
    tc = load_yaml("configs/teacher.yaml")
    fc = tc["d3a_filter"]
    _, _, rt = _cfgs()
    title = (
        pl.read_parquet(resolve(tc["out_dir"]) / "queries.parquet")
        .filter(pl.col("task") == "bitext_he2en")
        .select(pl.col("query_ref").alias("ref"), "title")
    )
    human = (
        pl.read_parquet(resolve(tc["out_dir"]) / "d1_align.parquet")
        .join(title, on="ref")
        .filter(~pl.col("title").str.contains(fc["liturgy_pattern"]))
    )
    thr = float(human["s_en2he"].quantile(fc["human_quantile"]))
    n = kept = 0
    for f in sorted(rt.glob("score_*block*.parquet")):
        d = pl.read_parquet(f).with_columns(keep=pl.col("s_src") >= thr)
        d.write_parquet(f)
        n, kept = n + d.height, kept + int(d["keep"].sum())
    stats = {"threshold": thr, "segments": n, "kept": kept, "kept_share": kept / n if n else None}
    (rt / "keep.stats.json").write_text(json.dumps(stats, indent=2))
    log.info("d3a filter: %s", stats)
    return stats


def _d3b_dir(section: str = "d3b") -> tuple[dict[str, Any], Any]:
    dc = load_yaml("configs/gen.yaml")[section]
    return dc, resolve(dc["out_dir"]) / "roundtrip"


def d3b_prepare(section: str = "d3b") -> dict[str, Any]:
    """Query tables of the D3b search queries (Hebrew and English) + teacher encoding jobs
    (<d3b out_dir>/roundtrip/jobs.yaml, for run_parallel.py), with the D3b training instruction."""
    from ein_mishpat_ai.gen.d3b import load as load_d3b

    tc = load_yaml("configs/teacher.yaml")
    dc, rt = _d3b_dir(section)
    rt.mkdir(parents=True, exist_ok=True)
    q = load_d3b(dc["out_dir"])
    prefix = f"Instruct: {dc['instruction']}\\nQuery: "
    jobs = []
    for lang in ("he", "en"):
        name = f"{section}_q_{lang}"
        q.select(pl.col("ref").alias("id"), pl.col(f"q_{lang}").alias("text")).write_parquet(
            rt / f"{name}.parquet"
        )
        for k in range(SHARDS_PER_BLOCK):
            cmd = (
                f"python scripts/encode_corpus.py {tc['model']} {rt.relative_to(resolve('.'))}/{name}.parquet "
                f"{tc['emb_dir']}/{tc['model']}/{name} --shard {k} --num-shards {SHARDS_PER_BLOCK} "
                f"--max-length {tc['max_length']['query']} --batch-size {tc['batch_size']} "
                f"--prefix {shlex.quote(prefix)}"
            )
            jobs.append({"name": f"{name}-s{k}", "cmd": cmd, "gpus": 1})
    (rt / "jobs.yaml").write_text(
        yaml.safe_dump(
            {"batch": f"{section.replace('d3b', 'd3b_roundtrip', 1)}_encode", "jobs": jobs},
            width=1000,
        )
    )
    return {"queries": q.height, "jobs": len(jobs)}


def d3b_mine(section: str = "d3b") -> dict[str, Any]:
    """Round trip + hard negatives for the D3b queries: teacher top-`mining.depth` over every
    training Hebrew document; `rank` = position of the query's own passage (0 = not in the top
    depth; the mix keeps rank ≤ d3b.keep_rank), negatives without false negatives as for D3a →
    roundtrip/neg_{he,en}.parquet: ref, rank, negatives, t_pos, t_neg."""
    from ein_mishpat_ai.data.download_sefaria import load_cfg
    from ein_mishpat_ai.eval.metrics import topk
    from ein_mishpat_ai.train.teacher import false_negative_filter

    tc = load_yaml("configs/teacher.yaml")
    mc = tc["mining"]
    _, rt = _d3b_dir(section)
    d_idx, d_mat = _emb(tc, "docs_he")
    pool_refs = [""] * len(d_idx)
    for r, i in d_idx.items():
        pool_refs[i] = r
    out = {}
    for lang in ("he", "en"):
        q_idx, q_mat = _emb({**tc, "out_dir": str(rt)}, f"{section}_q_{lang}")
        refs = [r for r in sorted(q_idx, key=q_idx.get) if r in d_idx]
        qv = q_mat[[q_idx[r] for r in refs]]
        top_s, top_i = topk(qv, d_mat, mc["depth"], chunk=max(256, int(4e9 // len(pool_refs))))
        rank = np.array(
            [
                next((k + 1 for k, j in enumerate(row) if pool_refs[j] == r), 0)
                for r, row in zip(refs, top_i, strict=True)
            ]
        )
        s_src = np.einsum(
            "nd,nd->n", qv.astype(np.float32), d_mat[[d_idx[r] for r in refs]].astype(np.float32)
        )
        keeps, removed = false_negative_filter(
            load_cfg(), mc, refs, [[r] for r in refs], top_s, top_i, pool_refs, s_src
        )
        pl.DataFrame(
            {
                "ref": refs,
                "rank": rank,
                "negatives": [[r for r, _ in k] for k in keeps],
                "t_pos": [[float(s)] for s in s_src],
                "t_neg": [[s for _, s in k] for k in keeps],
            },
            schema={
                "ref": pl.String,
                "rank": pl.Int32,
                "negatives": pl.List(pl.String),
                "t_pos": pl.List(pl.Float32),
                "t_neg": pl.List(pl.Float32),
            },
        ).write_parquet(rt / f"neg_{lang}.parquet")
        out[lang] = {
            "queries": len(refs),
            **{
                f"own_in_top{k}": float(((rank > 0) & (rank <= k)).mean()) for k in (1, 10, 50, 200)
            },
            "negatives_mean": float(np.mean([len(k) for k in keeps])),
            "removed": dict(removed),
        }
    (rt / "mine.stats.json").write_text(json.dumps(out, indent=2))
    log.info("d3b round trip + mining: %s", out)
    return out
