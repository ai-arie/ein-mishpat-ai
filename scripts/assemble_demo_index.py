"""Assemble the demo index from the encoded shards: release/space/index/{embeddings.npy (float16, L2-normalised,
row-aligned), passages.parquet (id, title, work, he, en, license_he, license_en), meta.json}.
usage: python scripts/assemble_demo_index.py"""

import json

import numpy as np
import polars as pl

from ein_mishpat_ai.utils.code_hash import code_hash
from ein_mishpat_ai.utils.io import now_iso, resolve

MODEL = "student-1.7b-full-d3b-bf16"
E = resolve(f"artifacts/emb/demo/{MODEL}")
D = resolve("release/space/index")
parts = sorted(E.glob("shard*_part*.npy"))
emb = np.concatenate([np.load(p) for p in parts]).astype(np.float16)
ids = pl.concat([pl.read_parquet(p.with_suffix(".parquet")) for p in parts])
idcol = "id" if "id" in ids.columns else ids.columns[0]
corpus = pl.read_parquet(D / "corpus.parquet").drop("text")
order = ids.select(pl.col(idcol).alias("id")).with_row_index("_row")
passages = order.join(corpus, on="id", how="left").sort("_row").drop("_row")
assert passages.height == emb.shape[0] == corpus.height and passages["he"].null_count() == 0
norms = np.linalg.norm(emb.astype(np.float32), axis=1)
assert np.allclose(norms, 1, atol=1e-2), (norms.min(), norms.max())
np.save(D / "embeddings.npy", emb)
passages.write_parquet(D / "passages.parquet", compression="zstd")
(D / "meta.json").write_text(json.dumps({
    "model": "ai-arie/ein-mishpat-ai-1.7b", "local_model": MODEL, "rows": int(emb.shape[0]), "dim": int(emb.shape[1]),
    "dtype": "float16", "query_prefix": "Instruct: Given a search query, retrieve Jewish texts that answer it or are key sources for it\nQuery: ",
    "document_prefix": "", "min_chars": 50, "built": now_iso(), "code_hash": code_hash(),
}, indent=2))
(D / "corpus.parquet").unlink()  # passages.parquet holds the same rows in index order
print(emb.shape, f"{emb.nbytes / 1e9:.2f} GB ->", D)
