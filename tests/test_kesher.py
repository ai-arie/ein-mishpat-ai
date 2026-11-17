"""KesherBench consistency: qrels point into the corpus, queries have positives, splits are
disjoint, and no machine/LLM-translated text is in any query or corpus (rule 4)."""

import json

import polars as pl
import pytest

from ein_mishpat_ai.utils.io import resolve

ROOT = resolve("data/bench/kesher")


@pytest.fixture(scope="module")
def tasks():
    if not ROOT.exists():
        pytest.skip("KesherBench not built")
    return [json.loads(f.read_text()) for f in sorted(ROOT.glob("*/task.json"))]


def test_qrels_in_corpus_and_queries_have_positives(tasks):
    for m in tasks:
        corpus = set(
            pl.read_parquet(ROOT / "corpora" / f"{m['corpus']}_he.parquet", columns=["id"])[
                "id"
            ].to_list()
        )
        for split in ("test", "dev"):
            q = pl.read_parquet(ROOT / m["task"] / f"qrels_{split}.parquet")
            assert set(q["corpus_id"].to_list()) <= corpus, (m["task"], split)
            with_pos = set(q["query_id"].to_list())
            for f in (ROOT / m["task"]).glob(f"queries_*_{split}.parquet"):
                ids = set(pl.read_parquet(f, columns=["id"])["id"].to_list())
                assert ids <= with_pos, f


def test_test_and_dev_queries_disjoint(tasks):
    for m in tasks:
        t = set(pl.read_parquet(ROOT / m["task"] / "qrels_test.parquet")["query_id"].to_list())
        d = set(pl.read_parquet(ROOT / m["task"] / "qrels_dev.parquet")["query_id"].to_list())
        assert not t & d, m["task"]


def test_no_synthetic_text(tasks):
    texts = pl.read_parquet(
        resolve("data/processed/texts.parquet"), columns=["ref", "lang", "synthetic"]
    )
    syn = {(r, lang) for r, lang, s in texts.iter_rows() if s}
    for f in ROOT.glob("corpora/*.parquet"):
        lang = f.stem.rsplit("_", 1)[1]
        ids = pl.read_parquet(f, columns=["id"])["id"].to_list()
        assert not {(i, lang) for i in ids} & syn, f
    for f in ROOT.glob("*/queries_*.parquet"):
        lang = f.stem.split("_")[1][:2]
        ids = pl.read_parquet(f, columns=["id"])["id"].to_list()
        assert not {(i, lang) for i in ids} & syn, f
