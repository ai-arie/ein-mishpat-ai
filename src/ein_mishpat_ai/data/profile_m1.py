"""reports/M1.md — data profile and validation summary."""

from __future__ import annotations

import json
from typing import Any

import polars as pl

from ein_mishpat_ai.data.refs import from_dotted
from ein_mishpat_ai.data.segments import TOKENIZER
from ein_mishpat_ai.utils.code_hash import code_hash
from ein_mishpat_ai.utils.io import now_iso, project_root, resolve

RESOLVED = ["ok", "passage"]
HAS_TEXT = ["ok", "passage", "no_segment", "unparseable"]
CORE_PREFIXES = (
    "Talmud > Bavli > Seder",
    "Mishnah > Seder",
    "Talmud > Yerushalmi > Seder",
    "Halakhah > Mishneh Torah > Sefer",
    "Tanakh > Torah",
    "Tanakh > Prophets",
    "Tanakh > Writings",
)
CORE_TITLE_PREFIXES = ("Rashi on ", "Tosafot on ", "Ramban on ")


def _md(df: pl.DataFrame, floats: int = 3) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:.{floats}f}")
            elif isinstance(v, int):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def core_titles(versions: pl.DataFrame) -> list[str]:
    he = versions.filter(pl.col("lang") == "he")
    mask = pl.lit(False)
    for p in CORE_PREFIXES:
        mask = mask | pl.col("categories").str.starts_with(p)
    for p in CORE_TITLE_PREFIXES:
        mask = mask | pl.col("title").str.starts_with(p)
    return he.filter(mask)["title"].unique().to_list()


def link_sides(links: pl.DataFrame) -> pl.DataFrame:
    return pl.concat(
        [
            links.select(
                pl.col(f"status{s}").alias("status"),
                pl.col(f"title{s}").alias("title"),
                pl.col(f"cat{s}").alias("cat"),
            )
            for s in ("1", "2")
        ]
    )


def validation(cfg: dict[str, Any]) -> dict[str, Any]:
    """Numbers the M1 tests assert on; also printed in the report."""
    p = cfg["processed"]
    versions = pl.read_parquet(resolve(p["versions"]))
    seg = pl.scan_parquet(resolve(p["segments"]))
    links = pl.read_parquet(
        resolve(p["links"]), columns=["status1", "status2", "title1", "title2", "cat1", "cat2"]
    )
    sides = link_sides(links).filter(pl.col("status").is_in(HAS_TEXT))
    core = sides.filter(pl.col("title").is_in(core_titles(versions)))

    def text_of(ref: str, lang: str) -> list[str]:
        return (
            seg.filter((pl.col("ref") == ref) & (pl.col("lang") == lang))
            .select("text_plain" if lang == "he" else "text_clean")
            .collect()
            .to_series()
            .to_list()
        )

    return {
        "link_sides_with_kept_text": sides.height,
        "resolved_rate_all": float(sides["status"].is_in(RESOLVED).mean()),
        "core_link_sides": core.height,
        "resolved_rate_core": float(core["status"].is_in(RESOLVED).mean()),
        "berakhot_2a1_he": text_of("Berakhot 2a:1", "he"),
        "genesis_1_1_en": text_of("Genesis 1:1", "en"),
        "mt_reading_shema_1_1_he": text_of("Mishneh Torah, Reading the Shema 1:1", "he"),
        "licenses": sorted(versions["license"].unique().to_list()),
    }


def bench_alignment(cfg: dict[str, Any]) -> dict[str, Any]:
    """Sefaria-bench refs vs our segments: how many resolve, and for Talmud whether the Wikisource
    segment at the same ref is the same passage (word overlap with the benchmark's Hebrew)."""
    from ein_mishpat_ai.data.normalize import clean, plain_hebrew

    bench = json.loads(
        (project_root() / "data/external/sefaria_bench/data/benchmark.json").read_text()
    )
    refs = [from_dotted(b["ref"]) for b in bench]
    seg = (
        pl.scan_parquet(resolve(cfg["processed"]["segments"]))
        .filter((pl.col("lang") == "he") & pl.col("ref").is_in(refs))
        .select("ref", "text_plain", "version_id")
        .collect()
    )
    found = set(seg["ref"].to_list())
    by_cat: dict[str, list[int]] = {}
    overlaps = []
    for b, r in zip(bench, refs, strict=True):
        by_cat.setdefault(b["category"], [0, 0])
        by_cat[b["category"]][0] += 1
        by_cat[b["category"]][1] += r in found
        if b["category"] == "Talmud" and r in found:
            ours = seg.filter(pl.col("ref") == r)["text_plain"].to_list()
            theirs = set(plain_hebrew(clean(b["he"])).split())
            best = max(len(theirs & set(o.split())) / max(1, len(theirs)) for o in ours)
            overlaps.append(best)
    return {
        "resolved": len(found),
        "total": len(refs),
        "by_category": by_cat,
        "talmud_overlap_mean": sum(overlaps) / max(1, len(overlaps)),
        "talmud_overlap_ge_0_5": sum(o >= 0.5 for o in overlaps) / max(1, len(overlaps)),
        "talmud_checked": len(overlaps),
    }


def tokens_per_word_sample() -> dict[str, float]:
    """Tokens per Hebrew/Aramaic word, first 50 segments of Berakhot (SETUP S3.4)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    df = (
        pl.scan_parquet(resolve("data/processed/segments.parquet"))
        .filter(
            (pl.col("ref").str.starts_with("Berakhot "))
            & (pl.col("title") == "Berakhot")
            & (pl.col("lang") == "he")
        )
        .sort("seg_order")
        .head(50)
        .collect()
    )
    words = sum(len(t.split()) for t in df["text_plain"].to_list())
    n_clean = sum(
        len(tok(t, add_special_tokens=False)["input_ids"]) for t in df["text_clean"].to_list()
    )
    n_plain = sum(
        len(tok(t, add_special_tokens=False)["input_ids"]) for t in df["text_plain"].to_list()
    )
    return {
        "words": words,
        "tokens_per_word_clean": n_clean / words,
        "tokens_per_word_plain": n_plain / words,
    }


def write_report(cfg: dict[str, Any]) -> None:
    p = cfg["processed"]
    versions = pl.read_parquet(resolve(p["versions"]))
    seg = pl.scan_parquet(resolve(p["segments"]))
    links = pl.read_parquet(resolve(p["links"]))
    cov = pl.read_parquet(resolve("data/meta/flatten_coverage.parquet"))
    L: list[str] = [
        f"# M1 — Data pipeline report\n\nGenerated {now_iso()} · code hash `{code_hash()}`\n"
    ]

    totals = seg.select(
        pl.len().alias("segments"),
        pl.col("ref").n_unique().alias("unique_refs"),
        pl.sum("n_tokens").alias("tokens"),
    ).collect()
    L += [
        "## Snapshot\n",
        f"- Kept versions: {versions.height:,} ({versions.filter(pl.col('lang') == 'he').height:,} Hebrew folder, {versions.filter(pl.col('lang') == 'en').height:,} English)",
        f"- Segments: {totals['segments'][0]:,} rows, {totals['unique_refs'][0]:,} unique refs, {totals['tokens'][0]:,} Qwen3 tokens",
        f"- Flattening: {cov['strings'].sum():,} non-empty text strings in the files → {cov['segments'].sum():,} segments ({cov.filter(pl.col('lost') != 0).height} version(s) differ; empty versions: {cov.filter(pl.col('strings') == 0).height})",
        f"- Licenses present: {', '.join(sorted(versions['license'].unique().to_list()))}",
        "",
    ]
    lic = (
        versions.group_by("license", "lang").agg(pl.len().alias("versions")).sort("license", "lang")
    )
    L += ["### Versions by license\n", _md(lic), ""]

    by_cat = (
        seg.group_by("category", "lang")
        .agg(
            pl.len().alias("segments"),
            pl.col("ref").n_unique().alias("refs"),
            pl.col("version_id").n_unique().alias("versions"),
            pl.sum("n_tokens").alias("tokens"),
        )
        .sort("category", "lang")
        .collect()
    )
    L += ["## Segments by category × language\n", _md(by_cat), ""]

    refs = (
        seg.group_by("ref", "category")
        .agg((pl.col("lang") == "he").any().alias("he"), (pl.col("lang") == "en").any().alias("en"))
        .collect()
    )
    covg = (
        refs.group_by("category")
        .agg(
            pl.col("he").sum().alias("refs_he"),
            pl.col("en").sum().alias("refs_en"),
            (pl.col("he") & pl.col("en")).sum().alias("both"),
        )
        .with_columns((pl.col("both") / pl.col("refs_he")).alias("share_he_with_en"))
        .sort("category")
    )
    L += [
        "## English coverage (refs with a kept Hebrew/Aramaic text that also have kept English)\n",
        "Shows where synthetic translations (D3a) are needed.\n",
        _md(covg),
        "",
    ]

    q = (
        seg.group_by("lang")
        .agg(
            pl.col("n_tokens").quantile(0.5).alias("p50"),
            pl.col("n_tokens").quantile(0.9).alias("p90"),
            pl.col("n_tokens").quantile(0.99).alias("p99"),
            (pl.col("n_tokens") > 256).mean().alias("share_gt_256"),
            (pl.col("n_tokens") > 512).mean().alias("share_gt_512"),
        )
        .sort("lang")
        .collect()
    )
    tpw = tokens_per_word_sample()
    he_all = (
        seg.filter(pl.col("lang") == "he").select(pl.sum("n_tokens"), pl.sum("n_words")).collect()
    )
    L += [
        "## Token lengths (Qwen3 tokenizer, text_clean)\n",
        _md(q),
        "",
        f"Tokens per Hebrew/Aramaic word: all Hebrew-folder segments {he_all['n_tokens'][0] / he_all['n_words'][0]:.2f} (text_clean incl. niqqud where present); Berakhot first 50 segments ({tpw['words']} words): {tpw['tokens_per_word_clean']:.2f} clean, {tpw['tokens_per_word_plain']:.2f} plain.",
        "",
    ]

    sides = link_sides(links)
    st = sides["status"].value_counts().sort("count", descending=True)
    val = validation(cfg)
    L += [
        "## Links\n",
        f"{links.height:,} link rows; {links['conn_type'].n_unique()} connection types. Each side resolved to segment refs (ranges > {p['max_expand']} segments kept as passage-level).\n",
        "### Side status\n",
        _md(st),
        "",
        f"- Resolved among sides whose work has kept text: **{val['resolved_rate_all']:.4f}** ({val['link_sides_with_kept_text']:,} sides). Below 0.99 because many works are kept only partially (e.g. only an English excerpt), not because of parse errors.",
        f"- Core works ({len(core_titles(versions))} titles: Bavli, Mishnah, Yerushalmi, Mishneh Torah, Tanakh, Rashi/Tosafot/Ramban): **{val['resolved_rate_core']:.4f}** of {val['core_link_sides']:,} sides — the parser check (≥ 0.99).",
        "- `unknown_node` is mostly dictionary entries (BDB, Klein, Jastrow: DictionaryNode, not flattened).",
        "",
    ]
    per_cat = (
        link_sides(links)
        .filter(pl.col("status").is_in(HAS_TEXT))
        .group_by("cat")
        .agg(pl.len().alias("sides"), pl.col("status").is_in(RESOLVED).mean().alias("resolved"))
        .sort("sides", descending=True)
    )
    L += ["### Resolution by category (sides with kept text)\n", _md(per_cat), ""]
    ct = (
        links.group_by("conn_type")
        .agg(
            pl.len().alias("rows"),
            ((pl.col("status1").is_in(RESOLVED)) & (pl.col("status2").is_in(RESOLVED)))
            .sum()
            .alias("both_resolved"),
        )
        .sort("rows", descending=True)
    )
    L += ["### Connection types\n", _md(ct), ""]
    pairs = (
        links.with_columns(
            pl.min_horizontal("cat1", "cat2").alias("a"),
            pl.max_horizontal("cat1", "cat2").alias("b"),
        )
        .group_by("conn_type", "a", "b")
        .agg(pl.len().alias("rows"))
        .sort("rows", descending=True)
        .head(40)
    )
    L += ["### Connection type × category pair (top 40)\n", _md(pairs), ""]

    ba = bench_alignment(cfg)
    L += [
        "## Sefaria benchmark refs vs our segments\n",
        f"{ba['resolved']:,} of {ba['total']:,} benchmark refs have a kept Hebrew/Aramaic segment at the same ref (needed for the leakage exclusion in M2).\n",
        _md(
            pl.DataFrame(
                [
                    {"category": k, "pairs": v[0], "resolved": v[1]}
                    for k, v in sorted(ba["by_category"].items())
                ]
            )
        ),
        "",
        f"Talmud: the Wikisource Bavli segment at the benchmark's ref shares on average {ba['talmud_overlap_mean']:.2f} of the benchmark Hebrew's words ({ba['talmud_overlap_ge_0_5']:.1%} of {ba['talmud_checked']} refs ≥ 0.5) — checks that Wikisource uses Sefaria's segmentation.",
        "",
    ]

    L += [
        "## Validation checks\n",
        f"- `Berakhot 2a:1` (he): {val['berakhot_2a1_he'][:1]}",
        f"- `Genesis 1:1` (en): {len(val['genesis_1_1_en'])} versions, e.g. {val['genesis_1_1_en'][:1]}",
        f"- `Mishneh Torah, Reading the Shema 1:1` (he): {len(val['mt_reading_shema_1_1_he'])} versions",
        f"- Licenses in versions table: {val['licenses']}",
        "- Unknown-license versions of editions that are NC elsewhere: excluded, list in `data/meta/excluded_unknown_editions.tsv`.",
        "",
    ]
    out = project_root() / "reports" / "M1.md"
    out.write_text("\n".join(L))
