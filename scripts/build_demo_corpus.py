"""Demo index corpus (2026-11-08): Tanakh, Mishnah, Bavli and Mishneh Torah base texts (no commentaries).

One row per passage: the Hebrew/Aramaic text is indexed (same text choice as KesherBench: data/processed/texts.parquet,
Hebrew text_plain, no synthetic rows; the Torah and the Five Megillot in the Masoretic text); the freely licensed English of the same ref, if any, is shown next to it.
Segments under 50 characters are left out (the short-fragment rule fixed on dev), except in the
Tanakh, where every verse is kept. Every row keeps
its license; English marked "unknown" is not included. Output: release/space/index/corpus.parquet (id, title, work, he, en, license_he, license_en).
usage: python scripts/build_demo_corpus.py
"""

import polars as pl
from masoretic import masoretic  # scripts/masoretic.py

from ein_mishpat_ai.utils.io import load_yaml, resolve

P = load_yaml("configs/data.yaml")["processed"]
WORKS = {  # category prefix -> label; base texts only
    "Tanakh > Torah": "Tanakh", "Tanakh > Prophets": "Tanakh", "Tanakh > Writings": "Tanakh",
    "Mishnah > Seder": "Mishnah", "Talmud > Bavli > Seder": "Bavli",
    **{f"Halakhah > Mishneh Torah > Sefer {s}": "Mishneh Torah" for s in
       ["Madda", "Ahavah", "Zemanim", "Nashim", "Kedushah", "Haflaah", "Zeraim", "Avodah", "Korbanot", "Taharah",
        "Nezikim", "Kinyan", "Mishpatim", "Shoftim"]},
}
t = pl.read_parquet(resolve(P["texts"]), columns=["ref", "lang", "title", "categories", "license", "synthetic", "text_clean", "text_plain"])
t = t.filter(~pl.col("synthetic"))
work = pl.lit(None, dtype=pl.String)
for pre, lab in WORKS.items():
    work = pl.when(pl.col("categories").str.starts_with(pre)).then(pl.lit(lab)).otherwise(work)
t = t.with_columns(work.alias("work")).filter(pl.col("work").is_not_null())
he = t.filter(pl.col("lang") == "he").select("ref", "title", "work", pl.col("text_plain").alias("he"), pl.col("license").alias("license_he"))
# The Torah and the Five Megillot: the Masoretic text, as in KesherBench (scripts/masoretic.py), not the modern-Hebrew
# rendering that the version rule of data/pairs.py prefers for them.
mt = masoretic().rename({"id": "ref"})
he = (he.join(mt, on="ref", how="left")
      .with_columns(pl.coalesce("text_mt", "he").alias("he"), pl.coalesce("license_mt", "license_he").alias("license_he"))
      .drop("text_mt", "license_mt"))
# English marked "unknown" is not shown in the demo (2026-11-08: such texts are not redistributed)
en = t.filter((pl.col("lang") == "en") & (pl.col("license") != "unknown")).select(
    "ref", pl.col("text_clean").alias("en"), pl.col("license").alias("license_en"))
assert he["ref"].is_unique().all() and en["ref"].is_unique().all()
# the 50-character rule does not apply to the Tanakh: every verse is searchable (decided 2026-11-12)
c = (he.filter((pl.col("he").str.len_chars() >= 50) | (pl.col("work") == "Tanakh")).join(en, on="ref", how="left").rename({"ref": "id"})
     .with_columns(pl.col("he").alias("text")))
out = resolve("release/space/index/corpus.parquet")
c.select("id", "text", "title", "work", "he", "en", "license_he", "license_en").write_parquet(out, compression="zstd")
print(c.group_by("work").agg(pl.len(), pl.col("en").is_not_null().sum().alias("with_english")).sort("work"))
print(c.group_by("license_he").agg(pl.len()), c.group_by("license_en").agg(pl.len()))
print("rows", c.height, "->", out)
