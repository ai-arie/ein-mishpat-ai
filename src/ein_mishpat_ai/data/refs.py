"""Sefaria refs: walking schemas, formatting segment addresses, parsing and expanding citations.

Addresses are tuples of *display* numbers (what appears in a ref): Integer-like types use the
1-based number (plus any `index_offsets_by_depth`), Talmud uses the array index + 1 (1a=1, 1b=2,
2a=3, …), so tuple order equals text order and parsing a citation gives the same tuple.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterator
from typing import Any

_TALMUD_TOKEN = re.compile(r"^(\d+)([ab])$")
_ADDR_TOKEN = re.compile(r"^\d+[ab]?(?::\d+[ab]?)*(?:-\d+[ab]?(?::\d+[ab]?)*)?$")


def primary_title(node: dict[str, Any], lang: str = "en") -> str:
    for t in node.get("titles", []):
        if t.get("lang") == lang and t.get("primary"):
            return t["text"]
    return node.get("title") or node.get("key") or ""


def format_section(addr_type: str, number: int) -> str:
    if addr_type == "Talmud":
        i = number - 1
        return f"{i // 2 + 1}{'ab'[i % 2]}"
    return str(number)


def parse_section(token: str) -> int:
    m = _TALMUD_TOKEN.match(token)
    if m:
        return (int(m.group(1)) - 1) * 2 + (1 if m.group(2) == "b" else 0) + 1
    return int(token)


def _offset(offsets: dict[str, Any] | None, depth: int, parent_idx: tuple[int, ...]) -> int:
    """index_offsets_by_depth: key = 1-based depth; value int or nested lists by parent index."""
    if not offsets or str(depth) not in offsets:
        return 0
    v = offsets[str(depth)]
    for i in parent_idx:
        if not isinstance(v, list):
            break
        v = v[i] if i < len(v) else 0
    return v if isinstance(v, int) else 0


def iter_segments(
    node_ref: str, node: dict[str, Any], text: Any
) -> Iterator[tuple[str, str, tuple[int, ...], str]]:
    """Yield (node_ref, ref, address, raw_text) for one jagged-array node."""
    types = node.get("addressTypes") or []
    offsets = node.get("index_offsets_by_depth")

    def rec(t: Any, idx: tuple[int, ...]) -> Iterator[tuple[str, str, tuple[int, ...], str]]:
        if isinstance(t, str):
            if not t.strip():
                return
            disp = tuple(i + 1 + _offset(offsets, d + 1, idx[:d]) for d, i in enumerate(idx))
            addr = ":".join(
                format_section(types[d] if d < len(types) else "Integer", n)
                for d, n in enumerate(disp)
            )
            yield node_ref, f"{node_ref} {addr}" if addr else node_ref, disp, t
        elif isinstance(t, list):
            for i, sub in enumerate(t):
                yield from rec(sub, idx + (i,))

    yield from rec(text, ())


def walk_version(
    index_title: str, schema: dict[str, Any], text: Any
) -> Iterator[tuple[str, str, tuple[int, ...], str]]:
    """Walk a version's text along the schema. Complex texts are dicts keyed by node title
    (default child: key ""). Ref of a nested node: "Index, Node, Child" (default nodes omitted)."""

    def rec(node: dict[str, Any], node_ref: str, t: Any):
        if "nodes" in node:
            if not isinstance(t, dict):
                return
            sibling_titles = {primary_title(c) for c in node["nodes"] if not c.get("default")}
            for child in node["nodes"]:
                if child.get("default"):
                    key, ref = "", node_ref
                else:
                    key = primary_title(child)
                    ref = f"{node_ref}, {key}"
                if key in t:
                    yield from rec(child, ref, t[key])
                elif child.get("key") in t and child["key"] not in sibling_titles:
                    # text keyed by the node's internal key; never borrow a sibling's text
                    yield from rec(child, ref, t[child["key"]])
        else:
            yield from iter_segments(node_ref, node, t)

    yield from rec(schema, index_title, text)


def split_citation(citation: str) -> tuple[str, str]:
    """'Rashi on Genesis 1:1:2' -> ('Rashi on Genesis', '1:1:2'); whole-node refs -> (ref, '')."""
    citation = citation.strip()
    head, _, last = citation.rpartition(" ")
    if head and _ADDR_TOKEN.match(last):
        return head, last
    return citation, ""


def parse_range(addr: str) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    if not addr:
        return (), ()
    try:
        start_s, _, end_s = addr.partition("-")
        start = tuple(parse_section(x) for x in start_s.split(":"))
        if not end_s:
            return start, start
        end_parts = tuple(parse_section(x) for x in end_s.split(":"))
        end = start[: max(0, len(start) - len(end_parts))] + end_parts
        return start, end
    except ValueError:
        return None


def expand(
    addresses: list[tuple[int, ...]], start: tuple[int, ...], end: tuple[int, ...]
) -> tuple[int, int]:
    """Index range [lo, hi) of sorted `addresses` covered by start..end (prefix semantics)."""
    lo = bisect_left(addresses, start)
    hi = bisect_right(addresses, end + (10**9,) * 8)
    return lo, max(lo, hi)


_DOTTED = re.compile(r"^(.*?)[.:]((?:\d+[ab]?[.:])*\d+[ab]?)$")


def from_dotted(ref: str) -> str:
    """Sefaria URL-style ref ('Rashi on Genesis.1.1:1', 'Iggeret HaRamban:1') -> 'Rashi on Genesis 1:1:1'."""
    m = _DOTTED.match(ref)
    if not m:
        return ref
    return f"{m.group(1)} {m.group(2).replace('.', ':')}"
