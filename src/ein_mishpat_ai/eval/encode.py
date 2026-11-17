"""Sharded, resumable encoding of a text table (project convention: float16 .npy shards + row-aligned refs).

Input: a parquet with columns `id` and `text`. Shard k of n takes rows k, k+n, k+2n, … after sorting
by text length (balanced shards); each shard is written in parts of `part_size` rows so a restart
resumes at the first missing part. Output dir: <out>/shard{k:02d}_part{j:04d}.{npy,parquet}.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ein_mishpat_ai.eval.encoder import Encoder, model_spec
from ein_mishpat_ai.utils.log import get_logger

log = get_logger("encode", None)


def encode_table(
    model: str,
    table: Path,
    out: Path,
    shard: int = 0,
    num_shards: int = 1,
    prefix: str = "",
    max_length: int = 512,
    batch_size: int = 64,
    part_size: int = 20_000,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(table, columns=["id", "text"])
    df = df.with_columns(pl.col("text").str.len_chars().alias("_len")).sort("_len", "id")
    df = df.with_row_index("_i").filter(pl.col("_i") % num_shards == shard).drop("_i", "_len")
    parts = [df.slice(i, part_size) for i in range(0, df.height, part_size)]
    todo = [
        j for j in range(len(parts)) if not (out / f"shard{shard:02d}_part{j:04d}.npy").exists()
    ]
    log.info(
        "%s shard %d/%d: %d rows, %d/%d parts to do",
        model,
        shard,
        num_shards,
        df.height,
        len(todo),
        len(parts),
    )
    if not todo:
        return
    enc = Encoder(model_spec(model, max_length=max_length, batch_size=batch_size))
    for j in todo:
        p = parts[j]
        emb = enc.encode(p["text"].to_list(), prefix=prefix).astype(np.float16)
        stem = out / f"shard{shard:02d}_part{j:04d}"
        p.select("id").write_parquet(f"{stem}.parquet")
        np.save(f"{stem}.tmp.npy", emb)
        Path(f"{stem}.tmp.npy").replace(f"{stem}.npy")  # the .npy marks the part as done
        log.info("part %d done (%d rows)", j, p.height)


def load_embeddings(out: Path) -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    mats = []
    for f in sorted(out.glob("shard*_part*.npy")):
        if f.name.endswith(".tmp.npy"):
            continue
        ids.extend(pl.read_parquet(f.with_suffix(".parquet"))["id"].to_list())
        mats.append(np.load(f))
    return ids, np.vstack(mats) if mats else np.zeros((0, 0), np.float16)
