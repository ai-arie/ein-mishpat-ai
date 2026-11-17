"""Links table: parse the links CSVs, resolve each citation to segment refs.

Status per side:
  ok            1..max_expand segments; `refs` lists them
  passage       more than max_expand segments; only first/last ref and the count are kept
  no_kept_text  the work exists (schema) but none of its versions is kept (license / language)
  no_segment    the work has kept text, but no kept segment falls in the cited range
  unknown_node  the cited title/node is not in any schema
  unparseable   the address could not be parsed
"""

from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ein_mishpat_ai.data.refs import expand, parse_range, primary_title, split_citation
from ein_mishpat_ai.utils.io import resolve
from ein_mishpat_ai.utils.log import get_logger

log = get_logger("links", "build_data.log")


def schema_node_refs(sefaria_dir) -> set[str]:
    out: set[str] = set()
    for f in glob.glob(str(sefaria_dir / "schemas" / "*.json")):
        try:
            s = json.load(open(f))
        except ValueError:
            continue

        def rec(node: dict[str, Any], ref: str) -> None:
            out.add(ref)
            for child in node.get("nodes", []):
                rec(child, ref if child.get("default") else f"{ref}, {primary_title(child)}")

        rec(s.get("schema", {}), s["title"])
    return out


class Resolver:
    def __init__(self, segments_path, schema_refs: set[str], max_expand: int):
        df = (
            pl.scan_parquet(segments_path)
            .select("node_ref", "address", "ref")
            .unique(subset=["ref"])
            .collect()
        )
        nodes: dict[str, list[tuple[tuple[int, ...], str]]] = defaultdict(list)
        for node, addr, ref in df.iter_rows():
            nodes[node].append((tuple(addr), ref))
        self.addrs: dict[str, list[tuple[int, ...]]] = {}
        self.refs: dict[str, list[str]] = {}
        for node, items in nodes.items():
            items.sort()
            self.addrs[node] = [a for a, _ in items]
            self.refs[node] = [r for _, r in items]
        self.children: dict[str, list[str]] = defaultdict(list)  # schema node -> descendant nodes
        for node in self.addrs:
            parts = node.split(", ")
            for i in range(1, len(parts)):
                self.children[", ".join(parts[:i])].append(node)
        self.schema_refs = schema_refs
        self.max_expand = max_expand
        log.info("resolver: %d nodes, %d segment refs", len(self.addrs), df.height)

    def resolve(self, citation: str) -> tuple[str, list[str], str, str, int]:
        citation = citation.strip()
        if citation in self.addrs or citation in self.children or citation in self.schema_refs:
            node, addr = citation, ""  # whole node; its name may end in a number ("…, Part 1")
        else:
            node, addr = split_citation(citation)
        rng = parse_range(addr)
        if rng is None:
            return "unparseable", [], "", "", 0
        if node in self.addrs:
            start, end = rng
            lo, hi = expand(self.addrs[node], start, end)
            refs = self.refs[node][lo:hi]
        elif not addr and node in self.children:  # whole complex node, e.g. "Guide…, Part 1"
            refs = [r for child in sorted(self.children[node]) for r in self.refs[child]]
        elif node in self.schema_refs:
            return "no_kept_text", [], "", "", 0
        else:
            return "unknown_node", [], "", "", 0
        n = len(refs)
        if n == 0:
            return "no_segment", [], "", "", 0
        if n > self.max_expand:
            return "passage", [], refs[0], refs[-1], n
        return "ok", refs, refs[0], refs[-1], n


def build_links(cfg: dict[str, Any], workers: int = 1) -> None:
    sdir = resolve(cfg["sefaria"]["out_dir"])
    p = cfg["processed"]
    res = Resolver(resolve(p["segments"]), schema_node_refs(sdir), p["max_expand"])
    files = sorted(
        glob.glob(str(sdir / "links" / "links*.csv")),
        key=lambda f: int("".join(c for c in f.rsplit("/", 1)[1] if c.isdigit()) or -1),
    )
    files = [f for f in files if f.rsplit("/", 1)[1][5:-4].isdigit()]
    cache: dict[str, tuple[str, list[str], str, str, int]] = {}
    cols: dict[str, list[Any]] = defaultdict(list)
    link_id = 0
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader)
            for row in reader:
                if len(row) < 7:
                    continue
                c1, c2, ctype, t1, t2, cat1, cat2 = row[:7]
                cols["link_id"].append(link_id)
                link_id += 1
                cols["citation1"].append(c1)
                cols["citation2"].append(c2)
                cols["conn_type"].append(ctype.strip() or "(empty)")
                cols["title1"].append(t1)
                cols["title2"].append(t2)
                cols["cat1"].append(cat1)
                cols["cat2"].append(cat2)
                for side, c in (("1", c1), ("2", c2)):
                    r = cache.get(c)
                    if r is None:
                        r = cache[c] = res.resolve(c)
                    status, refs, first, last, n = r
                    cols[f"status{side}"].append(status)
                    cols[f"refs{side}"].append(refs)
                    cols[f"first{side}"].append(first)
                    cols[f"last{side}"].append(last)
                    cols[f"n{side}"].append(n)
        log.info("links: %s done, %d rows so far, %d unique citations", f, link_id, len(cache))
    out = resolve(p["links"])
    pq.write_table(pa.table(dict(cols)), out, compression="zstd", row_group_size=500_000)
    log.info("links: %d rows -> %s", link_id, out)
