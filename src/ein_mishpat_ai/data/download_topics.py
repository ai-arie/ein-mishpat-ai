"""Sefaria topic pages with their source refs (setup notes S5). Raw responses, resumable."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ein_mishpat_ai.data.http import FileRateLimiter, fetch
from ein_mishpat_ai.utils.io import append_jsonl, atomic_write_bytes, now_iso, resolve
from ein_mishpat_ai.utils.log import get_logger

log = get_logger("download_topics", "download_topics.log")


def topic_paths(cfg: dict[str, Any]) -> tuple[Path, Path, Path]:
    out = resolve(cfg["topics"]["out_dir"])
    return out, out / "topics_list.json", out / "by_slug"


def slug_file(by_slug: Path, slug: str) -> Path:
    safe = re.sub(r"[^\w\-.]", "_", slug)[:200]
    return by_slug / f"{safe}.json"


def load_topic_list(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    s = cfg["sefaria"]
    out, list_path, _ = topic_paths(cfg)
    if list_path.exists():
        return json.loads(list_path.read_bytes())
    limiter = FileRateLimiter(s["api_rate_per_sec"])
    r = fetch(
        f"{s['api_base']}/topics",
        user_agent=s["user_agent"],
        retries=s["retries"],
        timeout=300,
        limiter=limiter,
        params={"limit": 0},
    )
    if r.status != 200 or r.content is None:
        raise RuntimeError(f"topics list failed: {r.error}")
    atomic_write_bytes(list_path, r.content)
    topics = json.loads(r.content)
    log.info("topics list: %d topics", len(topics))
    return topics


def run_topics(cfg: dict[str, Any]) -> dict[str, int]:
    s = cfg["sefaria"]
    out, _, by_slug = topic_paths(cfg)
    topics = load_topic_list(cfg)
    slugs = [t["slug"] for t in topics if t.get("slug")]
    todo = [sl for sl in slugs if not slug_file(by_slug, sl).exists()]
    log.info("topics: %d total, %d to fetch", len(slugs), len(todo))
    limiter = FileRateLimiter(s["api_rate_per_sec"])
    errors = 0
    t0 = time.time()
    for i, slug in enumerate(todo, 1):
        url = f"{s['api_base']}/v2/topics/{quote(slug, safe='')}"
        r = fetch(
            url,
            user_agent=s["user_agent"],
            retries=s["retries"],
            limiter=limiter,
            params={"with_refs": 1},
        )
        if r.status == 200 and r.content is not None:
            atomic_write_bytes(slug_file(by_slug, slug), r.content)
        else:
            errors += 1
            append_jsonl(
                out / "errors.jsonl",
                {"slug": slug, "status": r.status, "error": r.error, "at": now_iso()},
            )
        if i % 100 == 0 or i == len(todo):
            el = time.time() - t0
            log.info(
                "progress %d/%d errors=%d eta=%.0fmin",
                i,
                len(todo),
                errors,
                el / i * (len(todo) - i) / 60,
            )
    return {"total": len(slugs), "fetched": len(todo) - errors, "errors": errors}
