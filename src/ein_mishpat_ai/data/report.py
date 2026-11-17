"""Pilot report (S4) and the full download verification (S7)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ein_mishpat_ai.data.download_sefaria import (
    Paths,
    load_books,
    local_json_path,
    pilot_entries,
    version_entries,
)
from ein_mishpat_ai.data.download_topics import slug_file, topic_paths
from ein_mishpat_ai.data.licenses import is_allowed, normalize_license
from ein_mishpat_ai.data.unknown_report import unknown_license_lines
from ein_mishpat_ai.utils.code_hash import code_hash
from ein_mishpat_ai.utils.io import now_iso, project_root, read_jsonl, sha256_file

SANITY_HEBREW = ["Genesis", "Mishnah Berakhot", "Berakhot", "Mishneh Torah, Reading the Shema"]


def last_records(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    last: dict[str, dict[str, Any]] = {}
    for rec in read_jsonl(Paths(cfg).manifest):
        last[rec["url"]] = rec
    return last


def _mb(n: float) -> str:
    return f"{n / 1e6:,.1f} MB"


def _md_table(header: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "\\|") for x in r) + " |")
    return "\n".join(out)


def license_table(recs: list[dict[str, Any]]) -> list[list[Any]]:
    c: Counter = Counter()
    for r in recs:
        if "license_norm" in r:
            c[(repr(r.get("license_raw")), r["license_norm"], r.get("license_source"))] += 1
    return [
        [raw, norm, "yes" if is_allowed(norm) else "no", src, n]
        for (raw, norm, src), n in sorted(c.items(), key=lambda x: -x[1])
    ]


def pilot_report(cfg: dict[str, Any]) -> Path:
    books = load_books(cfg)
    entries = pilot_entries(cfg, version_entries(books))
    last = last_records(cfg)
    recs = [last[e["json_url"]] for e in entries if e["json_url"] in last]
    fields: Counter = Counter()
    for r in recs:
        fields.update(r.get("fields", []))
    status = Counter(r["status"] for r in recs)
    reasons = Counter(r.get("reason") or "kept" for r in recs)
    api = sum(r.get("license_source") == "api" for r in recs)
    fallback_tried = sum(
        r.get("license_source") in ("api", "missing") or (r.get("license_norm") == "unknown")
        for r in recs
    )
    fetched_bytes = sum(r.get("bytes", 0) for r in recs)
    kept_bytes = sum(r.get("bytes", 0) for r in recs if r["status"] == "kept")
    sizes = (
        json.loads(Paths(cfg).bucket_sizes.read_text()) if Paths(cfg).bucket_sizes.exists() else {}
    )
    all_entries = version_entries(books)
    total_bytes = sum(sizes.get(_obj_name(e["json_url"]), 0) for e in all_entries)
    times = sorted(r["fetched_at"] for r in recs if r.get("fetched_at"))

    lines = [
        "# S4 pilot report — license-filtered Sefaria download",
        "",
        f"Generated {now_iso()} · code hash `{code_hash()}` · books.json generated_at "
        f"{books.get('generated_at')}",
        "",
        f"Pilot entries: {len(entries)} · processed: {len(recs)} · status: {dict(status)}",
        f"Reasons: {dict(reasons)}",
        f"License from API fallback: {api} of {fallback_tried} versions with missing/unknown "
        "license in the file",
        "",
        "## Fields present in version JSON (count of pilot files)",
        "",
        ", ".join(f"`{k}` ({v})" for k, v in sorted(fields.items(), key=lambda x: (-x[1], x[0]))),
        "",
        "## Distinct raw license strings",
        "",
        _md_table(["raw", "normalised", "allowed", "source", "n"], license_table(recs)),
        "",
        "## Versions",
        "",
        _md_table(
            [
                "title",
                "folder",
                "versionTitle",
                "actualLanguage",
                "license_raw",
                "status",
                "reason",
                "size",
            ],
            [
                [
                    r["title"],
                    r["language"],
                    r.get("versionTitle", r["versionTitle_index"])[:70],
                    r.get("actualLanguage"),
                    r.get("license_raw"),
                    r["status"],
                    r.get("reason", ""),
                    _mb(r.get("bytes", 0)),
                ]
                for r in sorted(recs, key=lambda r: (r["title"], r["language"], r["status"]))
            ],
        ),
        "",
        "## Size projection",
        "",
        f"Pilot fetched {_mb(fetched_bytes)}, kept {_mb(kept_bytes)} "
        f"({kept_bytes / max(fetched_bytes, 1):.0%} of bytes).",
        f"All {len(all_entries):,} non-merged version files in the bucket: {_mb(total_bytes)} "
        f"(from the bucket listing). At the pilot's kept ratio: ~{_mb(total_bytes * kept_bytes / max(fetched_bytes, 1))} "
        "on disk (the pilot over-represents Genesis/Talmud, so treat this as rough).",
        f"Pilot time window: {times[0] if times else '-'} → {times[-1] if times else '-'}",
    ]
    out = project_root() / "reports" / "S_pilot.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


def _obj_name(json_url: str) -> str:
    from urllib.parse import unquote, urlparse

    return unquote(urlparse(json_url).path).split("/", 2)[2]  # strip "/<bucket>/"


def topics_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    _, list_path, by_slug = topic_paths(cfg)
    if not list_path.exists():
        return {"listed": 0, "downloaded": 0}
    topics = json.loads(list_path.read_bytes())
    downloaded = 0
    ge5 = 0
    ge5_curated = 0
    gen_by: Counter = Counter()
    for t in topics:
        f = slug_file(by_slug, t["slug"])
        if not f.exists():
            continue
        downloaded += 1
        d = json.loads(f.read_bytes())
        refs = set()
        curated = set()
        for group in (d.get("refs") or {}).values():
            for r in group.get("refs", []) if isinstance(group, dict) else []:
                if r.get("is_sheet"):
                    continue
                refs.add(r.get("ref"))
                gen_by[r.get("generatedBy") or "-"] += 1
                if not r.get("generatedBy"):
                    curated.add(r.get("ref"))
        ge5 += len(refs) >= 5
        ge5_curated += len(curated) >= 5
    return {
        "listed": len(topics),
        "downloaded": downloaded,
        "ge5_nonsheet_refs": ge5,
        "ge5_refs_without_generatedBy": ge5_curated,
        "generatedBy_counts": dict(gen_by),
    }


def verify(cfg: dict[str, Any]) -> tuple[Path, list[str]]:
    """S7: data/manifest.json + reports/S_setup_download.md. Returns (report path, failed checks)."""
    root = project_root()
    p = Paths(cfg)
    books = load_books(cfg)
    entries = version_entries(books)
    last = last_records(cfg)
    recs = list(last.values())
    failures: list[str] = []

    status = Counter(r["status"] for r in recs)
    by_lang_cat: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in recs:
        by_lang_cat[(r["language"], (r.get("categories") or ["?"])[0])][r["status"]] += 1

    # files on disk + manifest
    files = []
    disk_bytes = 0
    for sub in ("json", "schemas", "links"):
        for f in sorted((p.out / sub).rglob("*")) if (p.out / sub).exists() else []:
            if f.is_file() and not f.name.endswith(".part"):
                size = f.stat().st_size
                disk_bytes += size
                files.append(
                    {"path": str(f.relative_to(root)), "bytes": size, "sha256": sha256_file(f)}
                )
    for name in ("books.json", "table_of_contents.json", "last_export.txt"):
        f = p.out / name
        if f.exists():
            files.append(
                {
                    "path": str(f.relative_to(root)),
                    "bytes": f.stat().st_size,
                    "sha256": sha256_file(f),
                }
            )

    # sanity checks
    kept = [r for r in recs if r["status"] == "kept"]
    for t in SANITY_HEBREW:
        if not any(r["title"] == t and r["language"] == "Hebrew" for r in kept):
            failures.append(f"no kept Hebrew version of {t}")
    if not any(r["title"] == "Genesis" and r["language"] == "English" for r in kept):
        failures.append("no kept English version of Genesis")
    sha_by_path = {f["path"]: f["sha256"] for f in files}
    for r in kept:
        fp = local_json_path(cfg, r["url"])
        rel = str(fp.relative_to(root))
        if rel not in sha_by_path:
            failures.append(f"kept file missing on disk: {rel}")
        elif sha_by_path[rel] != r["sha256"]:
            failures.append(f"sha256 mismatch: {rel}")
    api_ok = {
        str(local_json_path(cfg, r["url"]).relative_to(root))
        for r in kept
        if r.get("license_source") in ("api", "override") and is_allowed(r["license_norm"])
    }
    for f in files:
        if f["path"].startswith(str((p.out / "json").relative_to(root))):
            if Path(f["path"]).name == "merged.json":
                failures.append(f"merged.json on disk: {f['path']}")
                continue
            lic = json.loads((root / f["path"]).read_bytes()).get("license")
            if not is_allowed(normalize_license(lic)) and f["path"] not in api_ok:
                failures.append(f"disallowed license on disk: {f['path']} ({lic!r})")
    meta = json.loads(p.meta_report.read_text()) if p.meta_report.exists() else {}
    if meta:
        bad_meta = {k: v for k, v in meta.get("status", {}).items() if k not in ("ok", "exists")}
        if bad_meta:
            failures.append(f"meta files not ok: {bad_meta}")
        if meta.get("schemas_on_disk") != meta.get("schemas_expected"):
            failures.append(
                f"schemas {meta.get('schemas_on_disk')} != {meta.get('schemas_expected')}"
            )
        numbered = {k: v for k, v in meta.get("links", {}).items() if k[5:-4].isdigit()}
        if not numbered or not all(v["header_ok"] for v in numbered.values()):
            failures.append("a links CSV has an unexpected header")
    else:
        failures.append("meta report missing")
    n_err = status.get("error", 0)
    if len(recs) < len(entries):
        failures.append(f"only {len(recs)} of {len(entries)} versions processed")
    if n_err > 0.005 * len(entries):
        failures.append(f"errors {n_err} > 0.5% of versions")

    # skipped English (what we leave out)
    skipped_en = [
        r
        for r in recs
        if r["language"] == "English"
        and r["status"] == "skipped"
        and str(r.get("reason", "")).startswith("license:")
        and r.get("actualLanguage") in ("en", None)
    ]
    by_cat_skipped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in skipped_en:
        by_cat_skipped[(r.get("categories") or ["?"])[0]].append(r)
    notable = [
        r
        for r in skipped_en
        if any(
            k in (r.get("versionTitle") or "") for k in ("William Davidson", "Touger", "Steinsaltz")
        )
    ]

    # English coverage per category
    titles_he: dict[str, set] = defaultdict(set)
    titles_en: dict[str, set] = defaultdict(set)
    for r in kept:
        cat = (r.get("categories") or ["?"])[0]
        (titles_he if r["language"] == "Hebrew" else titles_en)[cat].add(r["title"])
    cov_rows = []
    for cat in sorted(set(titles_he) | set(titles_en)):
        both = titles_he[cat] & titles_en[cat]
        cov_rows.append(
            [
                cat,
                len(titles_he[cat] | titles_en[cat]),
                len(titles_en[cat]),
                len(both),
                len(titles_he[cat] - titles_en[cat]),
            ]
        )

    topics = topics_summary(cfg)
    hf_rows = _hf_rows(cfg)

    manifest = {
        "generated_at": now_iso(),
        "code_hash": code_hash(),
        "snapshot": {
            "books_generated_at": books.get("generated_at"),
            "last_export": meta.get("last_export"),
        },
        "files": files,
    }
    (root / "data" / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))

    fetched = sorted(r["fetched_at"] for r in recs if r.get("fetched_at"))
    lines = [
        "# Setup and download report (S7)",
        "",
        f"Generated {now_iso()} · code hash `{code_hash()}`",
        "",
        "## Snapshot",
        "",
        f"- books.json generated_at: {books.get('generated_at')}; last_export.txt: {meta.get('last_export')}",
        f"- downloads: {fetched[0] if fetched else '-'} → {fetched[-1] if fetched else '-'}",
        "",
        "## Sanity checks",
        "",
        "All passed." if not failures else "\n".join(f"- **FAIL** {x}" for x in failures[:50]),
        "",
        "## Versions",
        "",
        f"Non-merged versions in index: {len(entries):,} · processed: {len(recs):,} · "
        f"{dict(status)} · bytes on disk (json+schemas+links): {_mb(disk_bytes)}",
        "",
        _md_table(
            ["folder", "top category", "kept", "skipped", "error"],
            [
                [lang, cat, c["kept"], c["skipped"], c["error"]]
                for (lang, cat), c in sorted(by_lang_cat.items())
            ],
        ),
        "",
        "Skip reasons: "
        + ", ".join(
            f"`{k}` {v}"
            for k, v in Counter(
                r.get("reason") for r in recs if r["status"] != "kept"
            ).most_common()
        ),
        "",
        "## License strings",
        "",
        _md_table(["raw", "normalised", "allowed", "source", "n"], license_table(recs)),
        "",
        "## English left out (skipped for license, English text)",
        "",
        f"{len(skipped_en)} versions, {_mb(sum(r.get('bytes', 0) for r in skipped_en))}. "
        "Full list: `reports/S_skipped_english.tsv`.",
        "",
        _md_table(
            ["top category", "versions", "size"],
            [
                [c, len(v), _mb(sum(r.get("bytes", 0) for r in v))]
                for c, v in sorted(by_cat_skipped.items(), key=lambda x: -len(x[1]))
            ],
        ),
        "",
        "William Davidson / Steinsaltz / Touger versions among them: "
        + (
            f"{len(notable)} (e.g. "
            + "; ".join(
                f"{r['title']} — {r.get('versionTitle')} ({r.get('license_raw')})"
                for r in notable[:5]
            )
            + ")"
            if notable
            else "none found"
        ),
        "",
        "## English coverage of kept texts per category (titles)",
        "",
        _md_table(
            ["top category", "titles", "with kept English", "with both", "Hebrew only"], cov_rows
        ),
        "",
        *unknown_license_lines(cfg, books, recs),
        "## Topics",
        "",
        json.dumps(topics, ensure_ascii=False),
        "",
        "## Models (Hugging Face cache)",
        "",
        _md_table(["repo", "on disk", "expected", "complete"], hf_rows),
    ]
    rep = root / "reports" / "S_setup_download.md"
    rep.write_text("\n".join(lines) + "\n")
    with (root / "reports" / "S_skipped_english.tsv").open("w") as f:
        f.write("category\ttitle\tversionTitle\tlicense_raw\tlicense_norm\tbytes\n")
        for r in sorted(skipped_en, key=lambda r: (r.get("categories") or ["?"], r["title"])):
            f.write(
                "\t".join(
                    str(x)
                    for x in [
                        " > ".join(r.get("categories") or []),
                        r["title"],
                        r.get("versionTitle"),
                        r.get("license_raw"),
                        r.get("license_norm"),
                        r.get("bytes", 0),
                    ]
                )
                + "\n"
            )
    return rep, failures


def hf_cache_dir(repo: str) -> Path:
    import os

    return (
        Path(os.environ.get("HF_HOME", project_root() / ".cache" / "huggingface"))
        / "hub"
        / ("models--" + repo.replace("/", "--"))
    )


def hf_expected_sizes(cfg: dict[str, Any]) -> dict[str, int]:
    cache = project_root() / ".tmp" / "hf_sizes.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from ein_mishpat_ai.data.download_hf import repo_size

    sizes = {}
    for repo in cfg["hf"]["models"]:
        try:
            sizes[repo] = repo_size(repo)
        except Exception:
            return sizes
    cache.write_text(json.dumps(sizes))
    return sizes


def hf_on_disk(repo: str) -> tuple[int, bool]:
    d = hf_cache_dir(repo) / "blobs"
    if not d.exists():
        return 0, False
    total = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
    incomplete = any(f.name.endswith(".incomplete") for f in d.iterdir())
    return total, not incomplete


def _hf_rows(cfg: dict[str, Any]) -> list[list[Any]]:
    exp = hf_expected_sizes(cfg)
    rows = []
    for repo in cfg["hf"]["models"]:
        have, no_partial = hf_on_disk(repo)
        e = exp.get(repo, 0)
        rows.append(
            [
                repo,
                f"{have / 1e9:.1f} GB",
                f"{e / 1e9:.1f} GB" if e else "?",
                "yes" if no_partial and e and have >= e else "no",
            ]
        )
    lb = project_root() / "data" / "external" / "leaderboard.json"
    rows.append(
        [
            "Sefaria/Rabbinic-Embedding-Leaderboard (leaderboard.json)",
            f"{lb.stat().st_size / 1e3:.0f} kB" if lb.exists() else "0",
            "-",
            "yes" if lb.exists() else "no",
        ]
    )
    return rows
