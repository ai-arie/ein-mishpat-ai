"""License-filtered download of the Sefaria export (setup notes S4).

Only versions with an allowed license (PD / CC0 / CC-BY / CC-BY-SA / Sefaria "unknown") are written to
disk. Every other version
is fetched into memory, recorded in the manifest with the reason, and discarded.
"""

from __future__ import annotations

import csv
import json
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from ein_mishpat_ai.data.http import FileRateLimiter, fetch
from ein_mishpat_ai.data.licenses import find_override, is_allowed, normalize_license
from ein_mishpat_ai.utils.io import (
    append_jsonl,
    atomic_write_bytes,
    load_yaml,
    now_iso,
    read_jsonl,
    resolve,
    sha256_bytes,
    sha256_file,
)
from ein_mishpat_ai.utils.log import get_logger

log = get_logger("download_sefaria", "download_sefaria.log")

LINKS_HEADER = [
    "Citation 1",
    "Citation 2",
    "Conection Type",
    "Text 1",
    "Text 2",
    "Category 1",
    "Category 2",
]
FOLDER_LANG = {"Hebrew": "he", "English": "en"}


class Paths:
    def __init__(self, cfg: dict[str, Any]):
        self.out = resolve(cfg["sefaria"]["out_dir"])
        self.books = self.out / "books.json"
        self.json_dir = self.out / "json"
        self.manifest = self.out / "versions_manifest.jsonl"
        self.meta_manifest = self.out / "meta_manifest.jsonl"
        self.meta_report = self.out / "meta_report.json"
        self.api_cache = self.out / "api_cache" / "versions"
        self.bucket_sizes = self.out / "bucket_sizes.json"


def load_cfg(path: str = "configs/data.yaml") -> dict[str, Any]:
    return load_yaml(path)


# ----------------------------------------------------------------------------- index + meta


def load_books(cfg: dict[str, Any], refresh: bool = False) -> dict[str, Any]:
    """books.json is fetched once and then always read from disk, so every run uses one snapshot."""
    p = Paths(cfg)
    if p.books.exists() and not refresh:
        return json.loads(p.books.read_bytes())
    s = cfg["sefaria"]
    r = fetch(s["books_index"], user_agent=s["user_agent"], retries=s["retries"])
    if r.status != 200 or r.content is None:
        raise RuntimeError(f"books.json download failed: {r.error}")
    data = json.loads(r.content)
    atomic_write_bytes(p.books, r.content)
    log.info(
        "saved books.json generated_at=%s entries=%d", data.get("generated_at"), len(data["books"])
    )
    return data


def bucket_sizes(cfg: dict[str, Any]) -> dict[str, int]:
    """Object sizes of json/ in the public bucket (GCS JSON API listing). Used only for progress %."""
    p = Paths(cfg)
    if p.bucket_sizes.exists():
        return json.loads(p.bucket_sizes.read_text())
    s = cfg["sefaria"]
    bucket = urlparse(s["bucket_base"]).path.strip("/")
    api = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    sizes: dict[str, int] = {}
    token = None
    while True:
        params = {"prefix": "json/", "maxResults": 1000, "fields": "items(name,size),nextPageToken"}
        if token:
            params["pageToken"] = token
        r = fetch(api, user_agent=s["user_agent"], retries=s["retries"], params=params)
        if r.status != 200 or r.content is None:
            log.warning("bucket listing failed (%s); progress will use file counts only", r.error)
            return {}
        page = json.loads(r.content)
        for it in page.get("items", []):
            sizes[it["name"]] = int(it["size"])
        token = page.get("nextPageToken")
        if not token:
            break
    p.bucket_sizes.write_text(json.dumps(sizes))
    log.info("bucket listing: %d objects under json/", len(sizes))
    return sizes


def _count_csv(path: Path) -> tuple[list[str], int]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        n = sum(1 for _ in reader)
    return header, n


def run_meta(cfg: dict[str, Any]) -> dict[str, Any]:
    s = cfg["sefaria"]
    p = Paths(cfg)
    books = load_books(cfg)
    specials = [f for f in books["special_files"] if not f["path"].startswith("misc/")]
    log.info("meta: %d special files (misc/ skipped)", len(specials))

    def one(f: dict[str, Any]) -> dict[str, Any]:
        dest = _safe_dest(p.out, f["path"])
        if dest.exists() and ("size" not in f or dest.stat().st_size == f["size"]):
            return {"path": f["path"], "status": "exists", "bytes": dest.stat().st_size}
        r = fetch(
            f["url"], user_agent=s["user_agent"], retries=s["retries"], timeout=s["timeout_sec"]
        )
        if r.status != 200 or r.content is None:
            return {"path": f["path"], "status": "error", "reason": r.error}
        size_ok = "size" not in f or len(r.content) == f["size"]
        atomic_write_bytes(dest, r.content)
        return {
            "path": f["path"],
            "status": "ok" if size_ok else "size_mismatch",
            "bytes": len(r.content),
            "expected": f.get("size"),
            "sha256": sha256_bytes(r.content),
            "fetched_at": now_iso(),
        }

    results = []
    with ThreadPoolExecutor(s["max_concurrency"]) as ex:
        for i, res in enumerate(as_completed([ex.submit(one, f) for f in specials]), 1):
            rec = res.result()
            results.append(rec)
            if rec["status"] not in ("ok", "exists"):
                log.warning("meta %s: %s", rec["path"], rec)
            append_jsonl(p.meta_manifest, rec)
            if i % 1000 == 0:
                log.info("meta: %d/%d", i, len(specials))

    links = {}
    for f in sorted((p.out / "links").glob("*.csv")):
        header, n = _count_csv(f)
        links[f.name] = {"rows": n, "header_ok": header == LINKS_HEADER, "header": header}
    report = {
        "generated_at": books.get("generated_at"),
        "last_export": (p.out / "last_export.txt").read_text().strip()
        if (p.out / "last_export.txt").exists()
        else None,
        "files": len(specials),
        "status": Counter(r["status"] for r in results),
        "schemas_on_disk": len(list((p.out / "schemas").glob("*.json"))),
        "schemas_expected": sum(f["path"].startswith("schemas/") for f in specials),
        "links": links,
    }
    p.meta_report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("meta done: %s", json.dumps({k: v for k, v in report.items() if k != "links"}))
    return report


# ----------------------------------------------------------------------------- version texts


def _safe_dest(root: Path, rel: str) -> Path:
    """root/rel, refusing anything that would escape root (paths come from a remote index)."""
    dest = (root / rel).resolve()
    if root.resolve() not in dest.parents:
        raise ValueError(f"unsafe path: {rel!r}")
    return dest


def local_json_path(cfg: dict[str, Any], json_url: str) -> Path:
    rel = unquote(urlparse(json_url).path).split("/json/", 1)[1]
    return _safe_dest(Paths(cfg).json_dir, rel)


def version_entries(books: dict[str, Any]) -> list[dict[str, Any]]:
    return [b for b in books["books"] if b["versionTitle"] != "merged"]


def pilot_entries(cfg: dict[str, Any], entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pc = cfg["sefaria"]["pilot"]
    chosen = [e for e in entries if e["title"] in set(pc["titles"])]
    taken_cats = {e["categories"][0] for e in chosen}
    rest = [e for e in entries if e["categories"] and e["categories"][0] not in taken_cats]
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for e in rest:
        by_cat.setdefault(e["categories"][0], []).append(e)
    rng = random.Random(pc["seed"])
    cats = sorted(by_cat)
    rng.shuffle(cats)
    extra: list[dict[str, Any]] = []
    while len(extra) < pc["n_random"] and any(by_cat[c] for c in cats):
        for c in cats:  # round robin so the random part spreads over categories
            if by_cat[c] and len(extra) < pc["n_random"]:
                extra.append(by_cat[c].pop(rng.randrange(len(by_cat[c]))))
    return chosen + extra


class VersionLicenseLookup:
    """Fallback for versions whose file has no usable license: /api/texts/versions/{title}, cached."""

    def __init__(self, cfg: dict[str, Any]):
        s = cfg["sefaria"]
        self.cfg = s
        self.cache_dir = Paths(cfg).api_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = FileRateLimiter(s["api_rate_per_sec"])

    def _versions(self, title: str) -> list[dict[str, Any]] | None:
        cache = self.cache_dir / (re.sub(r"[^\w\-. ,]", "_", title)[:200] + ".json")
        if cache.exists():
            return json.loads(cache.read_text())
        url = f"{self.cfg['api_base']}/texts/versions/{quote(title)}"
        r = fetch(
            url,
            user_agent=self.cfg["user_agent"],
            retries=self.cfg["retries"],
            limiter=self.limiter,
        )
        if r.status != 200 or r.content is None:
            return None
        data = json.loads(r.content)
        if not isinstance(data, list):
            return None
        # Keep metadata only (no text) in the cache.
        slim = [{k: v.get(k) for k in ("versionTitle", "language", "license")} for v in data]
        atomic_write_bytes(cache, json.dumps(slim, ensure_ascii=False).encode())
        return slim

    def lookup(self, title: str, version_title: str, lang: str) -> str | None:
        versions = self._versions(title)
        if not versions:
            return None
        for v in versions:
            if v.get("versionTitle") == version_title and v.get("language") == lang:
                lic = v.get("license")
                return lic if isinstance(lic, str) and lic.strip() else None
        return None


def title_language_tag(version_title: str) -> str | None:
    """Trailing language tag in a versionTitle: "La Bible … [fr]" → "fr"."""
    m = re.search(r"\[([A-Za-z]{2,3})\]\s*$", version_title or "")
    return m.group(1).lower() if m else None


def _needs_fallback(raw: object) -> bool:
    return not isinstance(raw, str) or normalize_license(raw) in ("unknown", "unrecognized")


def process_entry(
    cfg: dict[str, Any], entry: dict[str, Any], lookup: VersionLicenseLookup
) -> tuple[dict[str, Any], bytes | None]:
    """Fetch one version into memory and decide. Returns (manifest record, bytes to write or None)."""
    s = cfg["sefaria"]
    folder = entry["language"]
    rec: dict[str, Any] = {
        "url": entry["json_url"],
        "title": entry["title"],
        "language": folder,
        "versionTitle_index": entry["versionTitle"],
        "categories": entry["categories"],
    }
    r = fetch(
        entry["json_url"],
        user_agent=s["user_agent"],
        retries=s["retries"],
        timeout=s["timeout_sec"],
    )
    rec["fetched_at"] = now_iso()
    if r.status != 200 or r.content is None:
        rec.update(status="error" if r.status in (0, 429) or r.status >= 500 else "skipped")
        rec["reason"] = f"http_{r.status}" if r.status else (r.error or "network_error")
        return rec, None
    try:
        doc = json.loads(r.content)
        if not isinstance(doc, dict):
            raise ValueError("not an object")
    except ValueError:
        rec.update(status="skipped", reason="parse_error", bytes=len(r.content))
        return rec, None

    flags: list[str] = []
    license_raw = doc.get("license")
    license_source = "file"
    version_title = doc.get("versionTitle") or entry["versionTitle"]
    if _needs_fallback(license_raw):
        api_lic = lookup.lookup(entry["title"], version_title, FOLDER_LANG.get(folder, ""))
        if api_lic is not None and not _needs_fallback(api_lic):
            license_raw, license_source = api_lic, "api"
        else:
            license_source = "file" if isinstance(license_raw, str) else "missing"
    norm = normalize_license(license_raw)
    override = find_override(
        cfg["licenses"].get("overrides") or [],
        entry["title"],
        version_title,
        folder,
        license_raw,
    )
    if override is not None:
        norm, license_source = override["license"], "override"
    actual = doc.get("actualLanguage")
    rec.update(
        versionTitle=version_title,
        versionSource=doc.get("versionSource"),
        license_raw=license_raw,
        license_norm=norm,
        license_source=license_source,
        actualLanguage=actual,
        isSource=doc.get("isSource"),
        fields=sorted(doc.keys()),
        bytes=len(r.content),
    )
    if not is_allowed(norm):
        rec.update(status="skipped", reason=f"license:{norm}")
        return rec, None
    if folder == "English":
        if actual is None:
            flags.append("lang_assumed")
        elif actual != "en":
            rec.update(status="skipped", reason=f"not_english:{actual}")
            return rec, None
        # Some translations carry actualLanguage "en" but a language tag in the title, e.g. "[tr]".
        tag = title_language_tag(version_title)
        if tag and tag != "en":
            rec.update(status="skipped", reason=f"not_english:tag_{tag}")
            return rec, None
    rec.update(status="kept", reason="", flags=flags, sha256=sha256_bytes(r.content))
    return rec, r.content


def _done_urls(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    last: dict[str, dict[str, Any]] = {}
    for rec in read_jsonl(Paths(cfg).manifest):
        last[rec["url"]] = rec
    done = {}
    overrides = cfg["licenses"].get("overrides") or []
    for url, rec in last.items():
        if rec["status"] == "skipped":
            # A version skipped for its license is fetched again once the license is allowed
            # (rule change or override).
            if rec["reason"].startswith("license:") and (
                is_allowed(normalize_license(rec.get("license_raw")))
                or find_override(
                    overrides,
                    rec["title"],
                    rec.get("versionTitle", ""),
                    rec["language"],
                    rec.get("license_raw"),
                )
            ):
                continue
            done[url] = rec
        elif rec["status"] == "kept":
            path = local_json_path(cfg, url)
            if path.exists() and path.stat().st_size == rec["bytes"]:
                done[url] = rec
    return done


def run_texts(cfg: dict[str, Any], pilot: bool, verify_sha: bool = False) -> Counter:
    s = cfg["sefaria"]
    p = Paths(cfg)
    books = load_books(cfg)
    entries = version_entries(books)
    if pilot:
        entries = pilot_entries(cfg, entries)
    done = _done_urls(cfg)
    if verify_sha:
        for url, rec in list(done.items()):
            if rec["status"] == "kept" and sha256_file(local_json_path(cfg, url)) != rec["sha256"]:
                del done[url]
    todo = [e for e in entries if e["json_url"] not in done]
    log.info(
        "texts(%s): %d versions, %d already done, %d to fetch",
        "pilot" if pilot else "all",
        len(entries),
        len(entries) - len(todo),
        len(todo),
    )
    lookup = VersionLicenseLookup(cfg)
    counts: Counter = Counter()
    mb = 0.0
    t0 = time.time()
    with ThreadPoolExecutor(s["max_concurrency"]) as ex:
        futs = {ex.submit(process_entry, cfg, e, lookup): e for e in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            e = futs[fut]
            try:
                rec, data = fut.result()
            except Exception as exc:  # keep going; the URL is retried on the next run
                rec = {
                    "url": e["json_url"],
                    "title": e["title"],
                    "language": e["language"],
                    "versionTitle_index": e["versionTitle"],
                    "categories": e["categories"],
                    "status": "error",
                    "reason": f"exception:{type(exc).__name__}:{exc}"[:300],
                    "fetched_at": now_iso(),
                }
                data = None
            if data is not None:
                dest = local_json_path(cfg, e["json_url"])
                rec["path"] = str(dest.relative_to(p.out))
                atomic_write_bytes(dest, data)
                mb += len(data) / 1e6
            del data
            append_jsonl(p.manifest, rec)
            counts[rec["status"]] += 1
            if i % 200 == 0 or i == len(todo):
                el = time.time() - t0
                eta = el / i * (len(todo) - i)
                log.info(
                    "progress %d/%d kept=%d skipped=%d errors=%d written=%.0fMB eta=%.0fmin",
                    i,
                    len(todo),
                    counts["kept"],
                    counts["skipped"],
                    counts["error"],
                    mb,
                    eta / 60,
                )
    log.info("texts done: %s", dict(counts))
    return counts


def status(cfg: dict[str, Any]) -> dict[str, Any]:
    p = Paths(cfg)
    books = load_books(cfg)
    total = len(version_entries(books))
    last: dict[str, dict[str, Any]] = {}
    for rec in read_jsonl(p.manifest):
        last[rec["url"]] = rec
    c = Counter(r["status"] for r in last.values())
    return {"versions_total": total, "processed": len(last), **c}
