"""Check that flattening keeps every non-empty text string of a version (schema walk losses)."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl

from ein_mishpat_ai.data.normalize import clean
from ein_mishpat_ai.utils.io import resolve


def _count(t: Any) -> int:
    if isinstance(t, str):
        return int(bool(clean(t)))
    if isinstance(t, list):
        return sum(_count(x) for x in t)
    if isinstance(t, dict):
        return sum(_count(x) for x in t.values())
    return 0


def _one(args: tuple[str, str, str]) -> tuple[str, int]:
    sdir, vid, path = args
    return vid, _count(json.loads((Path(sdir) / path).read_bytes())["text"])


def coverage(cfg: dict[str, Any], workers: int = 48) -> pl.DataFrame:
    """Per version: strings in the file vs segments in the table."""
    sdir = str(resolve(cfg["sefaria"]["out_dir"]))
    p = cfg["processed"]
    versions = pl.read_parquet(resolve(p["versions"]))
    with ProcessPoolExecutor(workers) as ex:
        counts = dict(
            ex.map(
                _one,
                [(sdir, v, path) for v, path in versions.select("version_id", "path").iter_rows()],
                chunksize=20,
            )
        )
    segs = (
        pl.scan_parquet(resolve(p["segments"]))
        .group_by("version_id")
        .agg(pl.len().alias("segments"))
        .collect()
    )
    return (
        versions.select("version_id", "title", "language", "version_title", "category")
        .with_columns(pl.col("version_id").replace_strict(counts, default=0).alias("strings"))
        .join(segs, on="version_id", how="left")
        .with_columns(pl.col("segments").fill_null(0))
        .with_columns((pl.col("strings") - pl.col("segments")).alias("lost"))
    )
