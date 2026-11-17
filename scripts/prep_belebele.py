"""Build the clean general-domain tasks of configs/eval/regression_clean.yaml from Belebele.

Per task: corpus = the 488 passages in corpus_lang (id = passage link), queries = the 900 questions in query_lang
(id = link#question_number), qrels = question -> its passage. Same Parquet layout as eval/regression.py prepare.
usage: python scripts/prep_belebele.py
"""

import json

import polars as pl
from huggingface_hub import hf_hub_download

from ein_mishpat_ai.utils.io import load_yaml, resolve

cfg = load_yaml("configs/eval/regression_clean.yaml")
src = cfg["source"]
langs = {lg for t in cfg["tasks"].values() for lg in (t["query_lang"], t["corpus_lang"])}
B = {
    lg: pl.read_ndjson(hf_hub_download(src["repo"], f"data/{lg}.jsonl", repo_type="dataset", revision=src["revision"]))
    for lg in sorted(langs)
}
for lg, b in B.items():
    assert b.select(pl.col("link").n_unique()).item() == b.select(pl.col("flores_passage").n_unique()).item(), lg
stats = {}
for name, t in cfg["tasks"].items():
    out = resolve(cfg["data_dir"]) / name
    out.mkdir(parents=True, exist_ok=True)
    c = B[t["corpus_lang"]].unique("link", keep="first", maintain_order=True)
    corpus = c.select(pl.col("link").alias("id"), pl.lit("").alias("title"), pl.col("flores_passage").alias("text"))
    q = B[t["query_lang"]].with_columns((pl.col("link") + "#" + pl.col("question_number").cast(pl.String)).alias("id"))
    queries = q.select("id", pl.col("question").alias("text"))
    qrels = q.select(pl.col("id").alias("query_id"), pl.col("link").alias("corpus_id"))
    corpus.write_parquet(out / "corpus.parquet")
    queries.write_parquet(out / "queries.parquet")
    qrels.write_parquet(out / "qrels.parquet")
    stats[name] = {"corpus": corpus.height, "queries": queries.height}
print(json.dumps(stats, indent=2))
