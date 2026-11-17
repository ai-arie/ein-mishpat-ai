"""D3b synthetic search queries (go for a 100k-passage pilot, 2026-11-03).

For each selected Hebrew training passage the generator writes one Hebrew and one English search
query that the passage answers — a natural question or a short topic phrase as typed into a search
box, without quoting the passage or naming any reference. They teach "short query → passage", the
shape behind KesherBench K5 (topic → sources) that translation training erodes.

select: passages from the teacher's training Hebrew pool (docs_he), never a KesherBench query or
        target of any split, sampled round-robin over works; batches of consecutive segments of one
        node (context for the model, one query pair per segment). → <out_dir>/selection.parquet
run:    blocks of `block_segments`, one request per batch (JSON answer); resumable (blocks are written
        whole, the response cache resumes inside a block). → blocks/, progress.jsonl
        Exit 3 (pause for approval) when failed requests or unparsable answers exceed the limits.

Synthetic: training only, never evaluated on, never published (project rule 4 (README)).
"""

from __future__ import annotations

import json
import random
import re
import time
from collections import defaultdict
from typing import Any

import polars as pl

from ein_mishpat_ai.gen.llm import LLM
from ein_mishpat_ai.utils.io import load_yaml, now_iso, resolve
from ein_mishpat_ai.utils.log import get_logger

log = get_logger("d3b", "gen.log")

SYSTEM = (
    "You help build a search engine for classical Jewish texts (Bible commentary, Mishnah, Talmud, "
    "Midrash, halakhah, responsa, Kabbalah, Hasidut, ethics, liturgy). The user gives numbered "
    "segments of one work, in order. For each segment write two search queries that a learner "
    "might type and that this segment answers or is a key source for: one in Hebrew and one in "
    "English. Each query is either a short natural question or a short topic phrase (2-8 words); "
    "vary between the two. Rules: the segment must really answer the query; do not quote or "
    "paraphrase distinctive wording of the segment; do not name the book, author, chapter, page "
    "or any reference; use the neighbouring segments only as context. Output only a JSON object "
    'mapping each segment number to its queries, e.g. {"1": {"he": "...", "en": "..."}, '
    '"2": {"he": "...", "en": "..."}}, with exactly one entry per segment.'
)


def messages(title: str, refs: list[str], texts: list[str]) -> list[dict[str, str]]:
    body = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts, 1))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Work: {title}\n\n{body}"},
    ]


def parse(content: str | None, n: int) -> list[tuple[str, str]] | None:
    """(he, en) per segment in order, or None unless exactly n entries with both queries."""
    if not content:
        return None
    s = content.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        d = json.loads(s[s.index("{") : s.rindex("}") + 1])
    except ValueError:
        return None
    if not isinstance(d, dict) or sorted(d) != sorted(str(i) for i in range(1, n + 1)):
        return None
    out = []
    for i in range(1, n + 1):
        x = d[str(i)]
        if not isinstance(x, dict):
            return None
        he, en = (str(x.get(k) or "").strip() for k in ("he", "en"))
        if not he or not en or not re.search(r"[א-ת]", he):
            return None
        out.append((he, en))
    return out


def select(cfg: dict[str, Any], section: str = "d3b") -> dict[str, Any]:
    dc = load_yaml("configs/gen.yaml")[section]
    tc = load_yaml("configs/teacher.yaml")
    out = resolve(dc["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    pool = pl.read_parquet(resolve(tc["out_dir"]) / "docs_he.parquet", columns=["id"])
    t = pl.read_parquet(
        resolve(cfg["processed"]["texts"]),
        columns=[
            "ref",
            "lang",
            "title",
            "category",
            "node_ref",
            "seg_order",
            "split",
            "leak",
            "text_clean",
            "n_words",
            "n_tokens",
        ],
    ).filter(pl.col("lang") == "he")
    bench: set[str] = set()
    for f in resolve("data/bench/kesher").glob("*/qrels_*.parquet"):
        q = pl.read_parquet(f, columns=["query_id", "corpus_id"])
        bench |= set(q["query_id"].to_list()) | set(q["corpus_id"].to_list())
    # passages an earlier D3b set already covered (d3b_ext: the pilot)
    done = set()
    if dc.get("exclude_dir"):
        done = set(
            pl.read_parquet(resolve(dc["exclude_dir"]) / "selection.parquet", columns=["ref"])[
                "ref"
            ].to_list()
        )
    he = (
        t.filter(
            pl.col("ref").is_in(pool["id"].implode())
            & (pl.col("split") == "train")
            & ~pl.col("leak")
            & ~pl.col("ref").is_in(list(bench))
            & ~pl.col("ref").is_in(list(done))
            & (pl.col("n_words") >= dc["min_words"])
            & (pl.col("n_tokens") <= dc["max_segment_tokens"])
            & ~pl.col("category").is_in(dc["exclude_categories"])
        )
        .sort("node_ref", "seg_order")
        .with_columns(
            (pl.col("seg_order").diff().over("node_ref").fill_null(1) > 1)
            .cum_sum()
            .over("node_ref")
            .alias("_run")
        )
    )
    bid, key, n, tok, ids = -1, None, 0, 0, []
    for node, run, nt in he.select("node_ref", "_run", "n_tokens").iter_rows():
        if (node, run) != key or n == dc["batch_size"] or tok + nt > dc["max_batch_tokens"]:
            bid, key, n, tok = bid + 1, (node, run), 0, 0
        n += 1
        tok += nt
        ids.append(bid)
    he = he.with_columns(pl.Series("batch", ids))
    by: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for b, title, n_ in (
        he.group_by("batch").agg(pl.first("title"), pl.len().alias("n")).sort("batch").iter_rows()
    ):
        by[title].append((b, n_))
    rng = random.Random(dc["seed"])
    works = sorted(by)
    rng.shuffle(works)
    for w in works:
        rng.shuffle(by[w])
    chosen, got = [], 0
    while got < dc["segments"] and any(by[w] for w in works):
        for w in works:
            if by[w] and got < dc["segments"]:
                b, n_ = by[w].pop()
                chosen.append(b)
                got += n_
    rng.shuffle(chosen)
    order = {b: i for i, b in enumerate(chosen)}
    sel = (
        he.filter(pl.col("batch").is_in(chosen))
        .with_columns(pl.col("batch").replace_strict(order).alias("order"))
        .sort("order", "seg_order")
        .select("order", "ref", "title", "category", pl.col("text_clean").alias("he"), "n_tokens")
    )
    sel.write_parquet(out / "selection.parquet")
    stats = {
        "eligible_segments": he.height,
        "selected_segments": sel.height,
        "batches": len(chosen),
        "works": sel["title"].n_unique(),
        "he_tokens": int(sel["n_tokens"].sum()),
        "by_category": dict(
            sel.group_by("category").len().sort("len", descending=True).iter_rows()
        ),
    }
    (out / "selection_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    log.info("d3b selection: %s", stats)
    return stats


def _wait_for_key(after: str) -> None:
    """Block until .env was changed after `after` (the API key is rotated before D3b,
    2026-11-03); the key itself is never read here."""
    from datetime import datetime

    t = datetime.fromisoformat(after).timestamp()
    env = resolve(".env")
    while env.stat().st_mtime <= t:
        log.info("d3b waiting for the rotated API key (.env unchanged since %s)", after)
        time.sleep(600)


def budget_check(
    progress: list[dict[str, Any]], total_requests: int, budget: int, flag_after: int = 3
) -> dict[str, Any]:
    """Token budget (2026-11-04). `stop`: the next block could pass the budget (largest block
    so far + 20% margin). `flag`: the projected total over all requests is above the budget, judged
    only after `flag_after` blocks so that one unusual block does not raise it."""
    spent = sum(p.get("tokens_all", 0) for p in progress)
    requests = sum(p.get("requests", 0) for p in progress)
    largest = max((p.get("tokens_all", 0) for p in progress), default=0)
    projected = round(spent / requests * total_requests) if requests else 0
    return {
        "spent": spent,
        "projected": projected,
        "budget": budget,
        "stop": spent + 1.2 * largest > budget,
        "flag": len(progress) >= flag_after and projected > budget,
    }


def _tell_user(text: str) -> None:
    """Into the status page: the log, and the top of "Waiting for you" (logs/status_notes.md)."""
    with resolve("logs/events.tsv").open("a") as f:
        f.write(f"{now_iso()}\t{text}\n")
    notes = resolve("logs/status_notes.md")
    lines = notes.read_text().splitlines()
    at = next((i for i, x in enumerate(lines) if x.startswith("- ")), len(lines))
    notes.write_text("\n".join([*lines[:at], f"- **{text}**", *lines[at:]]) + "\n")


def run(section: str = "d3b") -> int:
    gc = load_yaml("configs/gen.yaml")
    dc = gc[section]
    if dc.get("wait_key_after"):
        _wait_for_key(dc["wait_key_after"])
    out = resolve(dc["out_dir"])
    (out / "blocks").mkdir(parents=True, exist_ok=True)
    llm = LLM(gc["cache"], concurrency=dc["concurrency"], task=dc["task"])
    params = {**gc["settings"]["no_think"], "max_tokens": dc["max_tokens"]}
    batches = (
        pl.read_parquet(out / "selection.parquet")
        .group_by("order", maintain_order=True)
        .agg("ref", "he", pl.first("title"))
    )
    prog_f = out / "progress.jsonl"
    progress = [json.loads(x) for x in prog_f.read_text().splitlines()] if prog_f.exists() else []
    done = len(progress)
    per = dc["block_batches"]
    budget = dc.get("budget_tokens")
    for k in range(done, (batches.height + per - 1) // per):
        if budget:
            b = budget_check(progress, batches.height, budget, dc["budget_flag_after_blocks"])
            if b["stop"]:
                (out / "budget_stop.json").write_text(json.dumps({**b, "time": now_iso()}))
                msg = (
                    f"D3b ({section}) stopped at the token budget: {b['spent']:,} tokens used, "
                    f"{k} blocks done, budget {budget:,}. Finished blocks stay usable."
                )
                log.warning(msg)
                _tell_user(msg)
                return 4
        part = batches.slice(k * per, per)
        cached = set(llm.cache)
        t0 = time.time()
        res = llm.map([(messages(t, r, h), params) for _, r, h, t in part.iter_rows()])
        rows, n_err, n_bad, tin, tout, new, tall = [], 0, 0, 0, 0, 0, 0
        for (_, refs, _, _), x in zip(part.iter_rows(), res, strict=True):
            if x.get("error"):
                n_err += 1
                continue
            # for the budget also answers from the cache: a crash inside a block must not hide them
            tall += (x.get("usage") or {}).get("total_tokens") or 0
            if x["key"] not in cached:
                new += 1
                tin += (x.get("usage") or {}).get("prompt_tokens") or 0
                tout += (x.get("usage") or {}).get("completion_tokens") or 0
            q = parse(x.get("content"), len(refs))
            if q is None:
                n_bad += 1
                continue
            rows += [
                {"ref": r, "q_he": he, "q_en": en, "key": x["key"]}
                for r, (he, en) in zip(refs, q, strict=True)
            ]
        if rows:
            pl.DataFrame(rows).with_columns(
                pl.lit(llm.model).alias("model"), pl.lit(True).alias("synthetic")
            ).write_parquet(out / "blocks" / f"block{k:03d}.parquet")
        s = {
            "block": k,
            "requests": part.height,
            "new_requests": new,
            "errors": n_err,
            "unparsed": n_bad,
            "segments": int(part["ref"].list.len().sum()),
            "with_queries": len(rows),
            "in_tokens": tin,
            "out_tokens": tout,
            "tokens_all": tall,
            "wall_s": round(time.time() - t0, 1),
            "retries": dict(llm.retried),
            "time": now_iso(),
        }
        llm.retried = {}
        with prog_f.open("a") as f:
            f.write(json.dumps(s) + "\n")
        log.info("d3b block %d: %s", k, s)
        progress.append(s)
        if budget and not (out / "budget_flag.json").exists():
            b = budget_check(progress, batches.height, budget, dc["budget_flag_after_blocks"])
            if b["flag"]:
                (out / "budget_flag.json").write_text(json.dumps({**b, "time": now_iso()}))
                msg = (
                    f"D3b ({section}) is heading above the token budget: projected "
                    f"{b['projected']:,} tokens, budget {budget:,} ({b['spent']:,} used so far). "
                    "It keeps running and stops by itself at the budget."
                )
                log.warning(msg)
                _tell_user(msg)
        if (
            n_err > dc["max_error_rate"] * part.height
            or n_bad > dc["max_unparsed_rate"] * part.height
        ):
            log.warning("d3b paused (needs approval): errors %d, unparsed %d", n_err, n_bad)
            return 3
    return 0


def load(out_dir: str | None = None) -> pl.DataFrame:
    """All D3b queries: ref, q_he, q_en (synthetic, training only)."""
    out = resolve(out_dir or load_yaml("configs/gen.yaml")["d3b"]["out_dir"])
    files = sorted((out / "blocks").glob("block*.parquet"))
    if not files:
        return pl.DataFrame(schema={"ref": pl.String, "q_he": pl.String, "q_en": pl.String})
    return pl.concat([pl.read_parquet(f, columns=["ref", "q_he", "q_en"]) for f in files])
