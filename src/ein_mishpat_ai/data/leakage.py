"""Benchmark leakage exclusion (project rule 1 (README)).

The 3,708 refs of Sefaria/Rabbinic-Hebrew-English-Pairs and their ±2 neighbouring segments never
enter training, in any language or as either side of a link. Benchmark refs use URL form and
older title spellings ("Bereishit Rabbah.1:1", "Rabbeinu Behaye, Bereshit.1:9") and are sometimes
section-level ("Ramban on Genesis 1:1" = every comment on the verse), so each ref is resolved via
schema title variants and expanded by prefix. Some works were restructured since the benchmark was
built (Rabbeinu Bahya, Akeidat Yitzchak: old depth-2 refs), so every match is verified by text: the
benchmark's Hebrew words must be ≥ 50% contained in the matched segments; otherwise the passage is
searched by word overlap inside the node. The result is widened by 2 segments on each side in the
node's text order (union of all versions and languages). If neither address nor text finds the
passage, the whole node is excluded. A ref whose node is absent from our data cannot leak.
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from typing import Any

import polars as pl

from ein_mishpat_ai.data.normalize import clean, plain_hebrew
from ein_mishpat_ai.data.refs import expand, from_dotted, parse_range, primary_title, split_citation
from ein_mishpat_ai.utils.io import resolve
from ein_mishpat_ai.utils.log import get_logger

log = get_logger("leakage", "build_data.log")
NEIGHBOURS = 2


def title_aliases(schemas_dir: str) -> dict[str, str]:
    """Any English title variant of a work or of one of its nodes -> canonical node ref."""
    alias: dict[str, str] = {}
    for f in glob.glob(f"{schemas_dir}/*.json"):
        try:
            s = json.load(open(f))
        except ValueError:
            continue
        title = s["title"]
        names = list(dict.fromkeys([title, *s.get("titleVariants", [])]))
        for n in names:
            alias.setdefault(n, title)

        def rec(node: dict[str, Any], ref: str, prefixes: list[str]) -> None:
            for c in node.get("nodes", []):
                if c.get("default"):
                    rec(c, ref, prefixes)
                    continue
                cref = f"{ref}, {primary_title(c)}"
                cnames = [t["text"] for t in c.get("titles", []) if t.get("lang") == "en"]
                combos = [f"{p}, {n}" for p in prefixes for n in cnames][:200]
                for a in combos:
                    alias.setdefault(a, cref)
                rec(c, cref, combos)

        rec(s.get("schema", {}), title, names[:20])
    return alias


def build_exclusion(cfg: dict[str, Any]) -> dict[str, Any]:
    p = cfg["processed"]
    bench = json.loads(resolve("data/external/sefaria_bench/data/benchmark.json").read_text())
    alias = title_aliases(str(resolve("data/raw/sefaria/schemas")))
    df = (
        pl.scan_parquet(resolve(p["segments"]))
        .select("node_ref", "address", "ref")
        .unique(subset=["ref"])
        .collect()
    )
    nodes: dict[str, list[tuple[tuple[int, ...], str]]] = defaultdict(list)
    for node, addr, ref in df.iter_rows():
        nodes[node].append((tuple(addr), ref))
    addrs = {}
    refs = {}
    for node, items in nodes.items():
        items.sort()
        addrs[node] = [a for a, _ in items]
        refs[node] = [r for _, r in items]

    bench_nodes = set()
    for b in bench:
        node, _ = split_citation(from_dotted(b["ref"]))
        bench_nodes.add(node if node in addrs else alias.get(node, node))
    words: dict[str, set[str]] = defaultdict(set)  # ref -> Hebrew words (all versions)
    he = (
        pl.scan_parquet(resolve(p["segments"]))
        .filter((pl.col("lang") == "he") & pl.col("node_ref").is_in(list(bench_nodes)))
        .select("ref", "text_plain")
        .collect()
    )
    for ref, text in he.iter_rows():
        words[ref] |= set(text.split())

    def contained(bw: set[str], seg_refs: list[str]) -> float:
        got: set[str] = set()
        for r in seg_refs:
            got |= words.get(r, set())
        return len(bw & got) / max(1, len(bw))

    def search(bw: set[str], node: str) -> tuple[int, int]:
        """Best contiguous run of segments whose words lie mostly inside the benchmark text."""
        best, span = 0.0, (0, 0)
        rs = refs[node]
        for i, r in enumerate(rs):
            w = words.get(r)
            if not w or len(w & bw) / len(w) < 0.6:
                continue
            j = i + 1
            while (
                j < len(rs)
                and j - i < 20
                and words.get(rs[j])
                and len(words[rs[j]] & bw) / len(words[rs[j]]) >= 0.6
            ):
                j += 1
            score = contained(bw, rs[i:j])
            if score > best:
                best, span = score, (i, j)
        return span if best >= 0.5 else (0, 0)

    rows: list[tuple[str, str, str]] = []  # (excluded ref, bench ref, how)
    stats: dict[str, int] = defaultdict(int)
    for b in bench:
        raw = from_dotted(b["ref"])
        node, addr = split_citation(raw)
        if node not in addrs:
            node = alias.get(node, node)
        if node not in addrs:
            stats["node_absent"] += 1
            rows.append(("", b["ref"], "node_absent"))
            continue
        rng = parse_range(addr)
        lo = hi = 0
        if rng is not None:
            lo, hi = expand(addrs[node], *rng)
        how = "exact" if hi - lo == 1 else "section"
        bw = set(plain_hebrew(clean(b["he"])).split())
        has_he = any(r in words for r in refs[node])
        if has_he and (hi <= lo or contained(bw, refs[node][lo:hi]) < 0.5):
            addr_span = (lo, hi)
            lo, hi = search(bw, node)
            how = "text_match"
            if hi <= lo and addr_span[1] > addr_span[0]:
                # address resolves but another edition's wording differs (e.g. William Davidson
                # vs Wikisource Bavli): keep the address match rather than the whole node
                (lo, hi), how = addr_span, "address_unverified"
        if hi <= lo:
            lo, hi, how = 0, len(addrs[node]), "whole_node"
        stats[how] += 1
        lo2, hi2 = max(0, lo - NEIGHBOURS), min(len(addrs[node]), hi + NEIGHBOURS)
        for i in range(lo2, hi2):
            rows.append((refs[node][i], b["ref"], how if lo <= i < hi else "neighbour"))
    ex = pl.DataFrame(rows, schema=["ref", "bench_ref", "how"], orient="row")
    out = resolve(p["leakage"])
    out.parent.mkdir(parents=True, exist_ok=True)
    ex.write_parquet(out)
    result = {
        "bench_refs": len(bench),
        **dict(stats),
        "excluded_segment_refs": ex.filter(pl.col("ref") != "")["ref"].n_unique(),
    }
    (out.parent / "leakage_stats.json").write_text(json.dumps(result, indent=2))
    log.info("leakage exclusion: %s", result)
    return result


def excluded_refs(cfg: dict[str, Any]) -> set[str]:
    ex = pl.read_parquet(resolve(cfg["processed"]["leakage"]))
    return set(ex.filter(pl.col("ref") != "")["ref"].to_list())
