"""'Unknown license' section of reports/S_setup_download.md. Reporting only: changes no decision."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ein_mishpat_ai.data.download_sefaria import Paths, title_language_tag
from ein_mishpat_ai.utils.io import project_root

ORIG_LANGS = {"he", "arc"}  # Hebrew-folder versions counted as Hebrew/Aramaic (plus missing)
MD_MAX_ROWS = 400  # longer lists are cut in the markdown; the TSV always has all rows

_YEAR = re.compile(r"\b(1[4-9]\d\d|20[0-2]\d)\b")
_PUBLISHER = re.compile(
    r"press|publish|publication|society|books|verlag|edition|library|company|\bco\.|sons|"
    r"bros|ltd|inc\b|house|seminary|university|typograph|druck|imprimerie|defus|dfus",
    re.I,
)


def _top(r: dict[str, Any]) -> str:
    return (r.get("categories") or ["?"])[0]


def _cat2(r: dict[str, Any]) -> str:
    return " > ".join((r.get("categories") or ["?"])[:2])


def _table(header: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append(
            "| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in r) + " |"
        )
    return "\n".join(out)


def _tsv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x).replace("\t", " ").replace("\n", " ") for x in r) + "\n")


def _cut(rows: list[list[Any]], tsv_name: str) -> tuple[list[list[Any]], str]:
    if len(rows) <= MD_MAX_ROWS:
        return rows, ""
    return rows[
        :MD_MAX_ROWS
    ], f"\n\n(First {MD_MAX_ROWS} of {len(rows)} rows; all in `reports/{tsv_name}`.)"


def is_orig(r: dict[str, Any]) -> bool:
    if r["language"] != "Hebrew" or r.get("actualLanguage") not in ORIG_LANGS | {None}:
        return False
    tag = title_language_tag(r.get("versionTitle") or r.get("versionTitle_index", ""))
    return tag in (None, "he", "arc")  # e.g. "Yiddish Translation [YI]" has actualLanguage "he"


def parse_year(*texts: str | None) -> int | None:
    """Earliest plausible year in the first text that has one (versionTitle before versionSource)."""
    for t in texts:
        years = [int(y) for y in _YEAR.findall(t or "")]
        if years:
            return min(years)
    return None


def parse_publisher(version_title: str | None, version_source: str | None) -> str:
    """Heuristic: a comma/semicolon part of versionTitle that looks like a publisher, else the
    versionSource domain. Marked as a guess in the report."""
    for part in re.split(r"[,;]|\s-\s|\s–\s", version_title or ""):
        if _PUBLISHER.search(part):
            return part.strip()
    src = version_source or ""
    if src.startswith("http"):
        return urlparse(src).netloc.removeprefix("www.")
    return src.strip()[:60]


def excluded_hebrew_works(recs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Works that have Hebrew/Aramaic versions but none of them kept → their versions."""
    by_title: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        if is_orig(r):
            by_title[r["title"]].append(r)
    return {t: v for t, v in by_title.items() if not any(x["status"] == "kept" for x in v)}


class TitleResolver:
    """Link CSV 'Text 1/2' → work title. Complex works appear as node titles ('Tur, Orach Chayim'),
    so trailing ', …' parts are dropped until a known work title matches."""

    def __init__(self, titles: set[str]):
        self.titles = titles
        self.cache: dict[str, str | None] = {}

    def __call__(self, text: str) -> str | None:
        if text in self.cache:
            return self.cache[text]
        t = text
        found = None
        while True:
            if t in self.titles:
                found = t
                break
            if ", " not in t:
                break
            t = t.rsplit(", ", 1)[0]
        self.cache[text] = found
        return found


def link_losses(
    cfg: dict[str, Any], books: dict[str, Any], lost: set[str], orig_titles: set[str]
) -> dict[str, Any]:
    resolve_title = TitleResolver({b["title"] for b in books["books"]})
    links_dir = Paths(cfg).out / "links"
    total = one = both = unresolved_sides = no_orig_sides = 0
    by_type: Counter = Counter()
    by_title: Counter = Counter()
    for f in sorted(links_dir.glob("links*.csv")):
        if not f.stem[5:].isdigit():  # links_by_book*.csv are aggregates, not link pairs
            continue
        with f.open(encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) < 5:
                    continue
                total += 1
                hit = 0
                for side in (row[3], row[4]):
                    w = resolve_title(side)
                    if w is None:
                        unresolved_sides += 1
                    elif w in lost:
                        hit += 1
                        by_title[w] += 1
                    elif w not in orig_titles:
                        no_orig_sides += 1
                if hit:
                    one += 1
                    both += hit == 2
                    by_type[row[2] or "(empty)"] += 1
    return {
        "total": total,
        "lose_one_or_more": one,
        "lose_both": both,
        "unresolved_sides": unresolved_sides,
        "sides_without_any_hebrew_version": no_orig_sides,
        "by_type": by_type,
        "by_title": by_title,
    }


def unknown_license_lines(
    cfg: dict[str, Any], books: dict[str, Any], recs: list[dict[str, Any]]
) -> list[str]:
    reports = project_root() / "reports"
    L = ["## Unknown license (allowed since 2026-11: Sefaria believes these are public domain)", ""]

    # (1) unknown versions by folder × top category
    unk = [r for r in recs if r.get("license_norm") == "unknown"]
    c: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in unk:
        foreign = (
            r["language"] == "English"
            and (
                r.get("actualLanguage") not in ("en", None)
                or title_language_tag(r.get("versionTitle", "")) not in (None, "en")
            )
        ) or (r["language"] == "Hebrew" and not is_orig(r))
        c[(r["language"], _top(r))]["foreign" if foreign else "same"] += 1
    rows = [
        [lang, cat, v["same"] + v["foreign"], v["same"], v["foreign"]]
        for (lang, cat), v in sorted(c.items(), key=lambda x: (x[0][0], -sum(x[1].values())))
    ]
    L += [
        "### 1. Versions with unknown license, by folder and top category",
        "",
        f"{len(unk):,} versions in total (license field missing or `unknown` in the file and not "
        "resolved by the API). *Same language* = English text in the English folder, Hebrew/Aramaic "
        "(or no actualLanguage) in the Hebrew folder; *other language* = translations into other "
        "languages (Yiddish, Persian, French, …).",
        "",
        _table(["folder", "top category", "unknown", "same language", "other language"], rows),
        "",
    ]

    # (2) works whose only Hebrew/Aramaic versions are excluded
    lost = excluded_hebrew_works(recs)
    orig_titles = {r["title"] for r in recs if is_orig(r)}
    loss_rows = []
    for title, vs in sorted(lost.items(), key=lambda x: (_cat2(x[1][0]), x[0])):
        for v in vs:
            loss_rows.append(
                [
                    title,
                    _cat2(v),
                    v.get("versionTitle", v.get("versionTitle_index")),
                    v.get("versionSource") or "",
                    v.get("license_raw"),
                    v["status"],
                    v.get("reason", ""),
                    v.get("bytes", 0),
                ]
            )
    hdr = [
        "title",
        "category",
        "versionTitle",
        "versionSource",
        "license_raw",
        "status",
        "reason",
        "bytes",
    ]
    _tsv(reports / "S_unknown_hebrew_losses.tsv", hdr, loss_rows)
    reasons = Counter(r.get("reason") or r["status"] for vs in lost.values() for r in vs)
    by_cat = Counter(_top(vs[0]) for vs in lost.values())
    md_rows, note = _cut(
        [
            [r[0], r[1], r[2][:80], r[3][:70], r[4], r[6], f"{r[7] / 1e3:,.0f} kB"]
            for r in loss_rows
        ],
        "S_unknown_hebrew_losses.tsv",
    )
    L += [
        "### 2. Works whose only Hebrew/Aramaic versions are excluded (coverage losses)",
        "",
        f"{len(lost):,} of {len(orig_titles):,} works with a Hebrew/Aramaic version have none kept "
        f"({sum(len(v) for v in lost.values()):,} versions, "
        f"{sum(r.get('bytes', 0) for vs in lost.values() for r in vs) / 1e6:,.1f} MB). "
        "Work level only: a kept version may still cover only part of a work (checked per segment in M1).",
        "",
        "Reasons: " + ", ".join(f"`{k}` {v}" for k, v in reasons.most_common()),
        "",
        "By top category (works): " + ", ".join(f"{k} {v}" for k, v in by_cat.most_common()),
        "",
        _table(
            ["title", "category", "versionTitle", "versionSource", "license", "reason", "size"],
            md_rows,
        )
        + note,
        "",
    ]

    # (3) English versions with unknown license, oldest first
    en_unk = [
        r
        for r in unk
        if r["language"] == "English"
        and r.get("actualLanguage") in ("en", None)
        and title_language_tag(r.get("versionTitle", "")) in (None, "en")
    ]
    en_rows = []
    for r in en_unk:
        vt, vs = r.get("versionTitle", r.get("versionTitle_index")), r.get("versionSource") or ""
        en_rows.append(
            [
                parse_year(vt, vs),
                parse_publisher(vt, vs),
                r["title"],
                _cat2(r),
                vt,
                vs,
                r.get("bytes", 0),
            ]
        )
    en_rows.sort(key=lambda x: (x[0] is None, x[0] or 0, x[2]))
    hdr = ["year", "publisher_guess", "title", "category", "versionTitle", "versionSource", "bytes"]
    _tsv(reports / "S_unknown_english.tsv", hdr, en_rows)
    n_year = sum(r[0] is not None for r in en_rows)
    n_old = sum(r[0] is not None and r[0] < 1930 for r in en_rows)
    md_rows, note = _cut(
        [
            [r[0] or "?", r[1][:50], r[2], r[3], r[4][:80], r[5][:60], f"{r[6] / 1e3:,.0f} kB"]
            for r in en_rows
        ],
        "S_unknown_english.tsv",
    )
    L += [
        "### 3. English versions with unknown license (oldest first)",
        "",
        f"{len(en_rows):,} versions, {sum(r[6] for r in en_rows) / 1e6:,.1f} MB. Year found for "
        f"{n_year:,}; {n_old:,} have a year before 1930. Year = earliest 4-digit year in versionTitle, "
        "else versionSource; publisher = a versionTitle part that looks like a publisher, else the "
        "source domain. Both are heuristic guesses; nothing here changes a license decision.",
        "",
        _table(
            [
                "year",
                "publisher (guess)",
                "title",
                "category",
                "versionTitle",
                "versionSource",
                "size",
            ],
            md_rows,
        )
        + note,
        "",
    ]

    # (4) link pairs losing a side
    ll = link_losses(cfg, books, set(lost), orig_titles)
    t = ll["total"] or 1
    L += [
        "### 4. Links that lose a side because of excluded Hebrew/Aramaic versions",
        "",
        f"Link rows (links0–16.csv): {ll['total']:,}. A side is *lost* if its work has Hebrew/Aramaic "
        "versions but none kept (section 2).",
        "",
        f"- at least one side lost: **{ll['lose_one_or_more']:,}** ({ll['lose_one_or_more'] / t:.1%})",
        f"- both sides lost: {ll['lose_both']:,} ({ll['lose_both'] / t:.1%})",
        f"- for context: sides whose work has no Hebrew/Aramaic version at all (not a license loss): "
        f"{ll['sides_without_any_hebrew_version']:,}; sides not resolved to a work title: "
        f"{ll['unresolved_sides']:,}",
        "",
        "By connection type (rows with ≥ 1 lost side):",
        "",
        _table(["type", "rows"], [[k, f"{v:,}"] for k, v in ll["by_type"].most_common(15)]),
        "",
        "Works causing the most lost link sides:",
        "",
        _table(["work", "lost sides"], [[k, f"{v:,}"] for k, v in ll["by_title"].most_common(25)]),
        "",
    ]
    return L
