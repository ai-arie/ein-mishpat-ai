"""project rules 1-2 (README): no Sefaria-bench segment (±2) and no held-out work in any training pair.

Integration tests run on the built data (skipped until it exists); unit tests always run."""

import json

import polars as pl
import pytest

from ein_mishpat_ai.data.splits import Splits
from ein_mishpat_ai.utils.io import load_yaml, resolve

P = load_yaml("configs/data.yaml")["processed"]


def _need(path):
    p = resolve(path)
    if not p.exists():
        pytest.skip(f"{path} not built")
    return p


@pytest.fixture(scope="module")
def splits():
    if not resolve("data/raw/sefaria/schemas").exists():
        pytest.skip("schemas not downloaded")
    return Splits()


@pytest.mark.parametrize(
    "title,node,expected",
    [
        ("Sanhedrin", "", "test"),
        ("Rashi on Sanhedrin", "", "test"),
        ("Mishneh Torah, Kings and Wars", "", "test"),
        ("Kessef Mishneh on Mishneh Torah, Kings and Wars", "", "test"),
        ("Rabbeinu Bahya", "Rabbeinu Bahya, Bamidbar", "test"),
        ("Rabbeinu Bahya", "Rabbeinu Bahya, Bereshit", None),
        ("Rav Hirsch on Torah, Numbers", "", "test"),
        ("Chidushei Agadot on Keritot", "", None),  # its schema wrongly lists Chullin as base
        ("Gittin", "", "dev"),
        ("Mishneh Torah, Sales", "", "dev"),
        ("Sifra", "", "dev"),
        ("Shabbat", "", None),
    ],
)
def test_split_rules(splits, title, node, expected):
    assert splits.split_of(title, node) == expected


def _training_refs() -> set[str]:
    d1 = pl.read_parquet(_need("data/train/d1_bitext.parquet"), columns=["ref"])
    d2 = pl.read_parquet(_need("data/train/d2_links.parquet"), columns=["a_refs", "b_refs"])
    refs = set(d1["ref"].to_list())
    refs |= set(d2["a_refs"].explode().to_list()) | set(d2["b_refs"].explode().to_list())
    return refs


@pytest.fixture(scope="module")
def training_refs():
    return _training_refs()


def test_every_bench_ref_resolved():
    ex = pl.read_parquet(_need(P["leakage"]))
    bench = json.loads(resolve("data/external/sefaria_bench/data/benchmark.json").read_text())
    resolved = set(ex.filter(pl.col("ref") != "")["bench_ref"].to_list())
    absent = set(ex.filter(pl.col("how") == "node_absent")["bench_ref"].to_list())
    assert {b["ref"] for b in bench} == resolved | absent
    # a ref we could not place must belong to a work we do not have at all
    assert not absent, sorted(absent)[:10]


def test_no_bench_segment_or_neighbour_in_training(training_refs):
    ex = pl.read_parquet(_need(P["leakage"]))
    leak = set(ex.filter(pl.col("ref") != "")["ref"].to_list())
    assert len(leak) > 3708
    hit = training_refs & leak
    assert not hit, sorted(hit)[:20]


def test_no_heldout_work_in_training(training_refs):
    texts = pl.read_parquet(_need(P["texts"]), columns=["ref", "split"])
    held = set(texts.filter(pl.col("split") != "train")["ref"].to_list())
    hit = training_refs & held
    assert not hit, sorted(hit)[:20]


def test_kesher_eval_refs_not_trained_as_pairs(training_refs):
    root = resolve("data/bench/kesher")
    if not root.exists():
        pytest.skip("KesherBench not built")
    d2 = pl.read_parquet(_need("data/train/d2_links.parquet"), columns=["a_refs", "b_refs"])
    trained = {(a, b) for x, y in d2.iter_rows() for a in x for b in y}
    trained |= {(b, a) for a, b in trained}
    for f in root.glob("*/qrels_*.parquet"):
        q = pl.read_parquet(f)
        pairs = set(zip(q["query_id"].to_list(), q["corpus_id"].to_list(), strict=True))
        assert not pairs & trained, f


def test_kesher_queries_not_in_training(training_refs):
    root = resolve("data/bench/kesher")
    if not root.exists():
        pytest.skip("KesherBench not built")
    q = set()
    for f in root.glob("*/qrels_*.parquet"):
        q |= set(pl.read_parquet(f, columns=["query_id"])["query_id"].to_list())
    hit = training_refs & q
    assert not hit, sorted(hit)[:20]


def test_teacher_tables_only_training_refs(training_refs):
    """M3a: every text the teacher encodes or mines from (queries, documents, negatives) is a
    training ref — negatives never come from benchmark, held-out or KesherBench query segments."""
    root = resolve("data/train/teacher")
    if not root.exists():
        pytest.skip("teacher tables not built")
    refs: set[str] = set()
    for f in [*root.glob("docs_*.parquet"), *root.glob("q_*.parquet")]:
        refs |= set(pl.read_parquet(f, columns=["id"])["id"].to_list())
    for f in (root / "negatives").glob("*.parquet"):
        refs |= set(pl.read_parquet(f, columns=["cands"])["cands"].explode().to_list())
    extra = refs - training_refs
    assert not extra, sorted(extra)[:20]


@pytest.mark.parametrize("mix", ["mix", "mix_d3b", "mix_d3b_all"])
def test_training_mix_only_training_refs(training_refs, mix):
    """M4: every ref in the training mix (queries, positives, negatives, texts) is a training ref."""
    root = resolve(f"data/train/{mix}")
    if not (root / "examples.parquet").exists():
        pytest.skip("training mix not built")
    ex = pl.read_parquet(root / "examples.parquet", columns=["query_ref", "positives", "negatives"])
    refs = set(ex["query_ref"].to_list())
    refs |= set(ex["positives"].explode().drop_nulls().to_list())
    refs |= set(ex["negatives"].explode().drop_nulls().to_list())
    refs |= set(pl.read_parquet(root / "texts.parquet", columns=["ref"])["ref"].to_list())
    extra = refs - training_refs
    assert not extra, sorted(extra)[:20]
    dev = set(pl.read_parquet(root / "dev_d1.parquet", columns=["ref"])["ref"].to_list())
    assert not dev & set(ex["query_ref"].to_list())
