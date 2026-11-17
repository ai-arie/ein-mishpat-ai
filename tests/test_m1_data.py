"""M1 validation. Skipped until data/processed exists."""

import polars as pl
import pytest

from ein_mishpat_ai.data.licenses import ALLOWED, RESTRICTIVE
from ein_mishpat_ai.data.normalize import clean, plain_hebrew
from ein_mishpat_ai.data.refs import from_dotted, parse_range, split_citation
from ein_mishpat_ai.utils.io import load_yaml, resolve

P = load_yaml("configs/data.yaml")["processed"]


def _need(path):
    p = resolve(path)
    if not p.exists():
        pytest.skip(f"{path} not built")
    return p


def test_normalize():
    raw = 'מתני׳ <big><strong>מאימתי</strong></big> קורין<sup class="footnote-marker">1</sup><i class="footnote">הערה <i>פנימית</i></i> את&nbsp;שמע'
    assert clean(raw) == "מתני׳ מאימתי קורין את שמע"
    assert plain_hebrew("בְּרֵאשִׁ֖ית בָּרָ֣א") == "בראשית ברא"
    assert plain_hebrew("כָּל־הָעָם {פ}") == "כל העם"


def test_ref_parsing():
    assert parse_range(split_citation("Berakhot 2a:1-3b:4")[1]) == ((3, 1), (6, 4))
    assert parse_range(split_citation("Genesis 1:1-5")[1]) == ((1, 1), (1, 5))
    assert from_dotted("Jerusalem Talmud Berakhot.1.3:1") == "Jerusalem Talmud Berakhot 1:3:1"
    assert from_dotted("Sefer HaIkkarim, Maamar 1.1:1") == "Sefer HaIkkarim, Maamar 1 1:1"


def test_versions_licenses_allowed_and_no_merged():
    v = pl.read_parquet(_need(P["versions"]))
    assert set(v["license"].unique()) <= ALLOWED
    assert not set(v["license"].unique()) & RESTRICTIVE
    assert not v["path"].str.ends_with("merged.json").any()
    assert v["version_id"].n_unique() == v.height


def test_segments_licenses_and_unique_refs():
    seg = pl.scan_parquet(_need(P["segments"]))
    lic = set(seg.select("license").unique().collect()["license"])
    assert lic <= ALLOWED
    dup = seg.group_by("version_id", "ref").len().filter(pl.col("len") > 1).collect()
    assert dup.height == 0


def test_known_refs():
    seg = pl.scan_parquet(_need(P["segments"]))

    def texts(ref, lang, col):
        return (
            seg.filter((pl.col("ref") == ref) & (pl.col("lang") == lang))
            .select(col)
            .collect()[col]
            .to_list()
        )

    assert any("מאימתי" in t[:20] for t in texts("Berakhot 2a:1", "he", "text_plain"))
    assert any("In the beginning" in t for t in texts("Genesis 1:1", "en", "text_clean"))
    assert texts("Mishneh Torah, Reading the Shema 1:1", "he", "text_clean")


def test_link_resolution_core_works():
    from ein_mishpat_ai.data.profile_m1 import validation

    _need(P["links"])
    v = validation({"processed": P})
    assert v["resolved_rate_core"] >= 0.99, v["resolved_rate_core"]


def test_synthetic_flag():
    from ein_mishpat_ai.data.synthetic import is_synthetic

    assert is_synthetic("Sefaria Kli Yakar, 2026 (Claude 3.7, ed. Francis Nataf)")
    assert is_synthetic("Google Translate")
    assert is_synthetic("x", "", "translated with DeepL")
    assert not is_synthetic("Tosefta Dem'ai", "Translation of Jean-Claude Sudre")
    seg = pl.scan_parquet(_need(P["segments"]))
    assert "synthetic" in seg.collect_schema().names()


def test_pilot_has_no_synthetic_text():
    task = resolve("data/bench/pilot_mt_bavli")
    if not task.exists():
        pytest.skip("pilot not built")
    v = pl.read_parquet(_need(P["versions"])).filter(pl.col("synthetic"))
    bad = set(v["version_title"].to_list())
    for f in ("queries_he.parquet", "queries_en.parquet"):
        q = pl.read_parquet(task / f)
        assert not set(q["version_title"].to_list()) & bad, f
