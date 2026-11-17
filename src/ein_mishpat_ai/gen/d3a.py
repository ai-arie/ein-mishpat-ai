"""D3a synthetic translations at scale (configs/gen.yaml `d3a`).

select: training-work Hebrew/Aramaic segments that are in link pairs and have no English version,
        grouped into batches of nearby segments of one node (gaps ≤ max_gap, ≤ size, ≤ max_batch_tokens), balanced
        round-robin over works inside two genre tiers, then shuffled so every block of the run is
        a representative sample. → data/gen/d3a/selection.parquet
run:    blocks of `block_segments`; one request per batch (JSON answer), single-segment requests
        for batches whose answer fails the format check. Concurrency is steered after each of
        `chunks_per_block` chunks (_steer; rules of 2026-11-01/02): step down on latency
        problems (each request slower per token than at 32 = we saturate the shared service),
        errors or retries; up one level after a fine chunk (~1k segments; 2026-11-02), ≥ 1 h
        after a step down; working hours ≤ 32, evenings and weekends ≤ 48. Changes →
        concurrency.jsonl. Pauses (exit 3) when projected output tokens or requests exceed the
        agreed envelope, or on errors/retries at the lowest level (a slowdown alone does not pause:
        16 is already the least load we put on the endpoint; 2026-11-02). Every block appends to progress.jsonl and reports/M3b_d3a_progress.md.

Synthetic: training only, never evaluated on, never published (project rule 4 (README)).
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import polars as pl

from ein_mishpat_ai.gen.llm import LLM
from ein_mishpat_ai.gen.translate import batch_messages, messages, parse_batch
from ein_mishpat_ai.utils.io import load_yaml, now_iso, resolve
from ein_mishpat_ai.utils.log import get_logger

log = get_logger("d3a", "gen.log")


def select(cfg: dict[str, Any]) -> dict[str, Any]:
    dc = load_yaml("configs/gen.yaml")["d3a"]
    out = resolve(dc["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t = pl.read_parquet(
        resolve(cfg["processed"]["texts"]),
        columns=[
            "ref",
            "lang",
            "title",
            "category",
            "categories",
            "node_ref",
            "seg_order",
            "split",
            "leak",
            "text_clean",
            "n_words",
            "n_tokens",
        ],
    )
    has_en = t.filter(pl.col("lang") == "en")["ref"].implode()
    eval_q: set[str] = set()
    for f in resolve("data/bench/kesher").glob("*/qrels_*.parquet"):
        eval_q |= set(pl.read_parquet(f, columns=["query_id"])["query_id"].to_list())
    d2 = pl.read_parquet(resolve("data/train/d2_links.parquet"), columns=["a_refs", "b_refs"])
    linked = set(d2["a_refs"].explode().to_list()) | set(d2["b_refs"].explode().to_list())
    he = t.filter(
        (pl.col("lang") == "he")
        & (pl.col("split") == "train")
        & ~pl.col("leak")
        & ~pl.col("ref").is_in(has_en)
        & ~pl.col("ref").is_in(list(eval_q))
        & pl.col("ref").is_in(list(linked))
        & pl.col("n_words").is_between(dc["min_words"], 10**9)
        & (pl.col("n_tokens") <= dc["max_segment_tokens"])
        & ~pl.col("category").is_in(dc["exclude_categories"])
    ).sort("node_ref", "seg_order")
    tier1 = pl.lit(False)
    for p in dc["tier1_prefixes"]:
        tier1 = tier1 | pl.col("categories").str.starts_with(p)
    he = he.with_columns(
        tier1.alias("tier1"),
        (pl.col("seg_order").diff().over("node_ref").fill_null(1) > dc["max_gap"])
        .cum_sum()
        .over("node_ref")
        .alias("_run"),
    )
    # batches: split each run of consecutive segments by count and token budget
    bid, cur_key, n, tok = -1, None, 0, 0
    ids = []
    for node, run, nt in he.select("node_ref", "_run", "n_tokens").iter_rows():
        if (node, run) != cur_key or n == dc["batch_size"] or tok + nt > dc["max_batch_tokens"]:
            bid, cur_key, n, tok = bid + 1, (node, run), 0, 0
        n += 1
        tok += nt
        ids.append(bid)
    he = he.with_columns(pl.Series("batch", ids))
    batches = he.group_by("batch").agg(pl.first("title"), pl.first("tier1"), pl.len().alias("n"))
    rng = random.Random(dc["seed"])
    chosen: list[int] = []
    stats: dict[str, Any] = {"eligible_segments": he.height, "eligible_batches": batches.height}
    for tier, budget in ((True, dc["tier1_segments"]), (False, dc["tier2_segments"])):
        by: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for b, title, n_ in (
            batches.filter(pl.col("tier1") == tier).select("batch", "title", "n").iter_rows()
        ):
            by[title].append((b, n_))
        works = sorted(by)
        rng.shuffle(works)
        for w in works:
            rng.shuffle(by[w])
        got = 0
        while got < budget and any(by[w] for w in works):
            for w in works:
                if by[w] and got < budget:
                    b, n_ = by[w].pop()
                    chosen.append(b)
                    got += n_
        stats[f"tier{'1' if tier else '2'}_segments"] = got
    rng.shuffle(chosen)
    order = {b: i for i, b in enumerate(chosen)}
    sel = (
        he.filter(pl.col("batch").is_in(chosen))
        .with_columns(pl.col("batch").replace_strict(order).alias("order"))
        .sort("order", "seg_order")
        .select(
            "order",
            "batch",
            "ref",
            "title",
            "category",
            "categories",
            "tier1",
            pl.col("text_clean").alias("he"),
            "n_tokens",
        )
    )
    sel.write_parquet(out / "selection.parquet")
    stats.update(
        selected_segments=sel.height,
        selected_batches=len(chosen),
        works=sel["title"].n_unique(),
        he_tokens=int(sel["n_tokens"].sum()),
        by_category=dict(sel.group_by("category").len().sort("len", descending=True).iter_rows()),
    )
    (out / "selection_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    log.info("d3a selection: %s", stats)
    return stats


def _block_summary(rows: list[dict[str, Any]], reqs: list[dict[str, Any]]) -> dict[str, Any]:
    secs = np.array(
        [r["seconds"] for r in reqs if r.get("seconds") is not None and not r["cached"]]
    )
    toks = np.array(
        [r["out_tokens"] for r in reqs if r.get("seconds") is not None and not r["cached"]]
    )
    per_tok = secs / np.maximum(toks, 1) if len(secs) else np.array([])
    return {
        "segments": len(rows),
        "translated": sum(r["translation"] is not None for r in rows),
        "requests_batched": sum(r["kind"] == "batch" for r in reqs),
        "requests_single": sum(r["kind"] == "single" for r in reqs),
        "batches_failed_format": sum(r["kind"] == "batch" and not r["ok"] for r in reqs),
        "request_errors": sum(r["error"] is not None for r in reqs),
        "cached": sum(bool(r["cached"]) for r in reqs),
        "in_tokens": int(sum(r["in_tokens"] or 0 for r in reqs)),
        "out_tokens": int(sum(r["out_tokens"] or 0 for r in reqs)),
        "latency_median_s": float(np.median(secs)) if len(secs) else None,
        "latency_p90_s": float(np.percentile(secs, 90)) if len(secs) else None,
        "s_per_out_token_median": float(np.median(per_tok)) if len(per_tok) else None,
    }


def _steer(
    conc: int,
    s: dict[str, Any],
    retried: dict[str, int],
    tok_s: float,
    st: dict[str, Any],
    dc: dict[str, Any],
    baseline: float | None,
    when: datetime,
) -> tuple[int, str, bool]:
    """Concurrency after one chunk → (next concurrency, reason, problem). Problem = errors, retries
    or each request slower per output token than `max_slowdown` × the first 32-parallel block (we
    saturate the shared service; `evening.max_slowdown` in the evening/night/weekend). Problem → one level down. Otherwise one level up after a fine
    chunk (2026-11-02), not within `cooldown_s` of a step down, up to `day_max` in working
    hours (weekdays outside `evening`) and the top level in the evening/weekend; back down if the
    step up did not raise output tokens/s by `min_gain`. Chunks with < `min_requests` new requests (resumed
    from the cache) change nothing. `st` carries last_down / upped_from between chunks."""
    levels, ev = dc["concurrency_levels"], dc["evening"]
    i = levels.index(conc)
    n_req = s["requests_batched"] + s["requests_single"] - s.get("cached", 0)
    spt = s["s_per_out_token_median"]
    evening = when.weekday() >= 5 or when.hour >= ev["from_utc"] or when.hour < ev["to_utc"]
    # evenings/nights tolerate a larger slowdown (2026-11-02 23:30: 32 at night)
    slow = ev.get("max_slowdown", dc["max_slowdown"]) if evening else dc["max_slowdown"]
    problem = (
        s["request_errors"] > dc["max_error_rate"] * max(n_req, 1)
        or sum(retried.values()) > dc["max_retry_rate"] * max(n_req, 1)
        or (baseline is not None and spt is not None and spt > slow * baseline)
    )
    cap = levels[-1] if evening else ev["day_max"]
    t = when.timestamp()
    new, why = conc, "keep"
    if conc > cap:
        new, why = cap, "down: working hours"
    elif n_req < ev["min_requests"]:
        return conc, "keep", False
    elif problem and i > 0:
        new, why = levels[i - 1], "down: endpoint slower per token, errors or retries"
        st["last_down"] = t
    elif st.get("upped_from") and tok_s < dc["min_gain"] * st["upped_from"]:
        new, why = levels[i - 1], "down: step up gave no throughput gain"
        st["last_down"] = t
    elif not problem and conc < cap and t - st.get("last_down", 0) >= ev["cooldown_s"]:
        new, why = levels[i + 1], "up: endpoint fine"
    st["upped_from"] = tok_s if why.startswith("up") else None
    return new, why, bool(problem)


def _translate(
    llm: LLM,
    part: pl.DataFrame,
    gl: bool,
    nm: bool,
    bparams: dict[str, Any],
    single: dict[str, Any],
    cached_before: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One request per batch (JSON answer); single requests for batches failing the format check."""
    res = llm.map(
        [(batch_messages(t, r, h, gl, c, nm), bparams) for _, r, h, t, c in part.iter_rows()]
    )
    rows, reqs, fallback = [], [], []
    for (o, refs, hes, title, cats), x in zip(part.iter_rows(), res, strict=True):
        parsed = parse_batch(x.get("content"), len(refs))
        reqs.append(_req("batch", x, parsed is not None, x.get("key") in cached_before))
        if parsed is None:
            fallback += [(o, r, h, title, cats) for r, h in zip(refs, hes, strict=True)]
        else:
            rows += [
                {"ref": r, "translation": tr, "mode": "batch", "order": o, "key": x["key"]}
                for r, tr in zip(refs, parsed, strict=True)
            ]
    if fallback:
        fres = llm.map([(messages(t, r, h, gl, c, nm), single) for _, r, h, t, c in fallback])
        for (o, r, _, _, _), x in zip(fallback, fres, strict=True):
            ok = bool(x.get("content")) and x.get("finish_reason") == "stop"
            reqs.append(_req("single", x, ok, x.get("key") in cached_before))
            rows.append(
                {
                    "ref": r,
                    "translation": x.get("content") if ok else None,
                    "mode": "single",
                    "order": o,
                    "key": x.get("key"),
                }
            )
    return rows, reqs


def run(cfg: dict[str, Any]) -> int:
    gc = load_yaml("configs/gen.yaml")
    dc = gc["d3a"]
    out = resolve(dc["out_dir"])
    (out / "blocks").mkdir(parents=True, exist_ok=True)
    sel = pl.read_parquet(out / "selection.parquet")
    prog_f = out / "progress.jsonl"
    prog = [json.loads(x) for x in prog_f.read_text().splitlines()] if prog_f.exists() else []
    single = gc["settings"]["no_think"]
    bparams = {**single, "max_tokens": dc["max_tokens_batch"]}
    llm = LLM(gc["cache"], task="d3a")
    levels = dc["concurrency_levels"]
    conc = prog[-1]["next_concurrency"] if prog else dc["start_concurrency"]
    conc = min(conc, levels[-1])  # levels may have been lowered since the last block
    steer: dict[str, Any] = {}  # _steer state: last step down, throughput before a step up
    baseline = next(
        (p["s_per_out_token_median"] for p in prog if p["concurrency"] == levels[1]), None
    )

    batches = sel.group_by("order", maintain_order=True).agg(
        "ref", "he", pl.first("title"), pl.first("categories")
    )
    seg_per_order = batches["ref"].list.len().to_numpy()
    starts = np.concatenate([[0], np.cumsum(seg_per_order)])
    total = int(starts[-1])
    blk = dc["block_segments"]
    nblocks = int(np.ceil(total / blk))
    for k in range(len(prog), nblocks):
        lo = int(np.searchsorted(starts, k * blk))
        hi = int(np.searchsorted(starts, min((k + 1) * blk, total)))
        part = batches.slice(lo, hi - lo)
        conc_start = conc
        retried: dict[str, int] = defaultdict(int)
        rows, reqs = [], []
        t0 = time.time()
        gl = dc.get("glossary_from_block") is not None and k >= dc["glossary_from_block"]
        nm = dc.get("numerals_from_block") is not None and k >= dc["numerals_from_block"]
        cuts = np.linspace(0, part.height, dc["chunks_per_block"] + 1).astype(int)
        for c in range(dc["chunks_per_block"]):
            llm.concurrency = conc
            llm.retried = {}
            cached_before = set(llm.cache)
            tc = time.time()
            r_, q_ = _translate(
                llm,
                part.slice(cuts[c], cuts[c + 1] - cuts[c]),
                gl,
                nm,
                bparams,
                single,
                cached_before,
            )
            rows += r_
            reqs += q_
            for key, v in llm.retried.items():
                retried[key] += v
            cs = _block_summary(r_, q_)
            new_out = sum(q["out_tokens"] or 0 for q in q_ if not q["cached"])
            tok_s = new_out / max(time.time() - tc, 1e-9)
            new, why, _ = _steer(
                conc, cs, llm.retried, tok_s, steer, dc, baseline, datetime.now(timezone.utc)
            )
            if new != conc:
                ch = {
                    "time": now_iso(),
                    "block": k,
                    "chunk": c,
                    "from": conc,
                    "to": new,
                    "why": why,
                    "s_per_out_token": cs["s_per_out_token_median"],
                    "baseline": baseline,
                    "out_tok_per_s": round(tok_s, 1),
                }
                with (out / "concurrency.jsonl").open("a") as f:
                    f.write(json.dumps(ch) + "\n")
                log.info("d3a concurrency: %s", ch)
                conc = new
        llm.retried = dict(retried)
        wall = time.time() - t0
        pl.DataFrame(rows).with_columns(
            pl.lit(llm.model).alias("model"),
            pl.lit(gl).alias("glossary"),
            pl.lit(nm).alias("numerals"),
            pl.lit(True).alias("synthetic"),
        ).write_parquet(out / "blocks" / f"block{k:03d}.parquet")
        s = _block_summary(rows, reqs)
        s.update(
            block=k,
            concurrency=conc_start,
            concurrency_end=conc,
            wall_s=round(wall, 1),
            retries=dict(llm.retried),
            time=now_iso(),
        )
        s["out_tok_per_s"] = round(s["out_tokens"] / wall, 1) if wall else None
        done_seg = sum(p["segments"] for p in prog) + s["segments"]
        done_out = sum(p["out_tokens"] for p in prog) + s["out_tokens"]
        done_req = sum(p["requests_batched"] + p["requests_single"] for p in prog) + (
            s["requests_batched"] + s["requests_single"]
        )
        s["projected_out_tokens"] = int(done_out / done_seg * total)
        s["projected_requests"] = int(done_req / done_seg * total)

        # concurrency rule
        n_req = s["requests_batched"] + s["requests_single"]
        failing = (
            s["request_errors"] > dc["max_error_rate"] * n_req
            or sum(llm.retried.values()) > dc["max_retry_rate"] * n_req
        )
        problem = failing or (
            baseline is not None
            and s["s_per_out_token_median"] is not None
            and s["s_per_out_token_median"] > dc["max_slowdown"] * baseline
        )
        if conc_start == conc == levels[1] and baseline is None:
            baseline = s["s_per_out_token_median"]
        # the chunks steer the concurrency (_steer); the block only reports the net change
        action = "up" if conc > conc_start else "down" if conc < conc_start else "keep"
        s.update(
            problem=bool(problem),
            action=action,
            next_concurrency=conc,
            done_segments=done_seg,
            total_segments=total,
        )
        reasons = {
            "projected output tokens above max_out_tokens": s["projected_out_tokens"]
            > dc["max_out_tokens"],
            "projected requests above max_requests": s["projected_requests"] > dc["max_requests"],
            "errors or retries at the lowest concurrency": failing and conc == levels[0],
            "pause_after_segments reached": bool(
                dc.get("pause_after_segments") and done_seg >= dc["pause_after_segments"]
            ),
        }
        pause = [r for r, hit in reasons.items() if hit]
        s["paused"] = bool(pause)
        s["pause_reasons"] = pause
        prog.append(s)
        with prog_f.open("a") as f:
            f.write(json.dumps(s) + "\n")
        _report(prog, dc)
        log.info("d3a block %d: %s", k, s)
        if pause:
            log.warning("d3a paused (needs approval): %s", "; ".join(pause))
            return 3
    return 0


def _req(kind: str, x: dict[str, Any], ok: bool, cached: bool) -> dict[str, Any]:
    u = x.get("usage") or {}
    return {
        "kind": kind,
        "ok": ok,
        "cached": cached,
        "error": x.get("error"),
        "seconds": x.get("seconds"),
        "in_tokens": u.get("prompt_tokens"),
        "out_tokens": u.get("completion_tokens"),
    }


def _report(prog: list[dict[str, Any]], dc: dict[str, Any]) -> None:
    tot = {
        k: sum(p[k] for p in prog)
        for k in (
            "segments",
            "translated",
            "requests_batched",
            "requests_single",
            "in_tokens",
            "out_tokens",
            "batches_failed_format",
            "request_errors",
            "wall_s",
        )
    }
    last = prog[-1]
    L = [
        "# M3b — D3a synthetic translations: progress\n",
        f"Updated {last['time']} · envelope: ≤ {dc['max_out_tokens'] / 1e6:.0f}M output tokens, "
        f"≤ {dc['max_requests']:,} requests (pause for approval beyond)\n",
        f"**Done {tot['segments']:,} / {last['total_segments']:,} segments** "
        f"({tot['translated']:,} translated) · requests {tot['requests_batched']:,} batched + "
        f"{tot['requests_single']:,} single · tokens in {tot['in_tokens'] / 1e6:.1f}M / out "
        f"{tot['out_tokens'] / 1e6:.1f}M · {tot['out_tokens'] / max(1, tot['segments']):.0f} "
        f"out tokens/segment · elapsed {tot['wall_s'] / 3600:.1f} h\n",
        f"Projected for the full run: {last['projected_out_tokens'] / 1e6:.0f}M output tokens, "
        f"{last['projected_requests']:,} requests; remaining ≈ "
        f"{(last['total_segments'] - tot['segments']) * (tot['wall_s'] / max(1, tot['segments'])) / 3600:.1f} h "
        "at the average speed so far.\n",
        "| block | segments | concurrency | requests (batch+single) | format fails | errors | "
        "retries | out tokens | out tok/s | latency median / p90 (s) | action |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in prog:
        L.append(
            f"| {p['block']} | {p['segments']:,} | {p['concurrency']} | "
            f"{p['requests_batched']:,}+{p['requests_single']:,} | {p['batches_failed_format']} | "
            f"{p['request_errors']} | {sum(p['retries'].values())} | {p['out_tokens'] / 1e6:.2f}M | "
            f"{p['out_tok_per_s']:.0f} | {p['latency_median_s'] or 0:.0f} / "
            f"{p['latency_p90_s'] or 0:.0f} | {p['action']}{' **PAUSED**' if p['paused'] else ''} |"
        )
    (resolve("reports") / "M3b_d3a_progress.md").write_text("\n".join(L) + "\n")


def redo(cfg: dict[str, Any]) -> dict[str, Any]:
    """Re-translate the batches of blocks before `numerals_from_block` that contain a Hebrew-letter
    numeral, with the current prompt (glossary + gematria digits); 2026-11-02 (option b).
    One file per block under redo/ (resumable by block; the cache resumes inside a block)."""
    from ein_mishpat_ai.gen.translate import gematria_digits

    gc = load_yaml("configs/gen.yaml")
    dc = gc["d3a"]
    out = resolve(dc["out_dir"])
    (out / "redo").mkdir(parents=True, exist_ok=True)
    single = gc["settings"]["no_think"]
    bparams = {**single, "max_tokens": dc["max_tokens_batch"]}
    # load on the shared endpoint: the lowest level in working hours, the top level otherwise (
    # 2026-11-02 rules; the redo is short, so no steering)
    now, ev = datetime.now(timezone.utc), dc["evening"]
    evening = now.weekday() >= 5 or now.hour >= ev["from_utc"] or now.hour < ev["to_utc"]
    conc = dc["concurrency_levels"][-1 if evening else 0]
    llm = LLM(gc["cache"], concurrency=conc, task="d3a_redo")
    batches = (
        pl.read_parquet(out / "selection.parquet")
        .group_by("order", maintain_order=True)
        .agg("ref", "he", pl.first("title"), pl.first("categories"))
    )
    starts = np.concatenate([[0], np.cumsum(batches["ref"].list.len().to_numpy())])
    blk = dc["block_segments"]
    stats = {}
    for k in range(dc["numerals_from_block"]):
        f = out / "redo" / f"block{k:03d}.parquet"
        if f.exists() or not (out / "blocks" / f"block{k:03d}.parquet").exists():
            continue
        lo = int(np.searchsorted(starts, k * blk))
        hi = int(np.searchsorted(starts, (k + 1) * blk))
        part = batches.slice(lo, hi - lo).filter(
            pl.col("he").map_elements(
                lambda hs: any(gematria_digits(h) != h for h in hs), return_dtype=pl.Boolean
            )
        )
        llm.retried = {}
        t0 = time.time()
        rows, reqs = _translate(llm, part, True, True, bparams, single, set(llm.cache))
        pl.DataFrame(rows).with_columns(
            pl.lit(llm.model).alias("model"),
            pl.lit(True).alias("glossary"),
            pl.lit(True).alias("numerals"),
            pl.lit(True).alias("synthetic"),
        ).write_parquet(f)
        s = _block_summary(rows, reqs)
        s.update(block=k, batches=part.height, wall_s=round(time.time() - t0, 1), time=now_iso())
        with (out / "redo" / "progress.jsonl").open("a") as fh:
            fh.write(json.dumps(s) + "\n")
        log.info("d3a redo block %d: %s", k, s)
        stats[k] = s
    return stats


COLS = [
    "ref",
    "translation",
    "mode",
    "order",
    "key",
    "model",
    "glossary",
    "numerals",
    "redo",
    "synthetic",
    "block",
]


def load(out_dir: str = "data/gen/d3a") -> pl.DataFrame:
    """All D3a translations; rows re-translated by `redo` replace the block's originals."""
    out = resolve(out_dir)
    parts = []
    for f in sorted((out / "blocks").glob("block*.parquet")):
        b = pl.read_parquet(f)
        for col in ("glossary", "numerals"):  # early blocks predate these columns
            if col not in b.columns:
                b = b.with_columns(pl.lit(False).alias(col))
        r = out / "redo" / f.name
        b = b.with_columns(pl.lit(False).alias("redo"))
        if r.exists():
            new = pl.read_parquet(r).with_columns(pl.lit(True).alias("redo"))
            b = pl.concat(
                [b.filter(~pl.col("ref").is_in(new["ref"].implode())), new.select(b.columns)]
            )
        parts.append(b.with_columns(pl.lit(int(f.stem[5:])).alias("block")).select(COLS))
    return pl.concat(parts, how="vertical_relaxed")
