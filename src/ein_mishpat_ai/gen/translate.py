"""Synthetic English translations (D3a). Training data only, never evaluated on, never
published (project rule 4 (README)); every output is tagged synthetic=true downstream.

Trial: 50 Hebrew/Aramaic training segments that have a free human English
translation, stratified by genre; each generator setting translates them and the teacher compares
the result with the human translation (data/gen/trial/).
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import polars as pl

from ein_mishpat_ai.gen.llm import LLM
from ein_mishpat_ai.utils.io import load_yaml, resolve
from ein_mishpat_ai.utils.log import get_logger

log = get_logger("translate", "gen.log")

SYSTEM = (
    "You are an expert translator of Hebrew and Aramaic Jewish texts: Bible, Mishnah, Talmud, "
    "Midrash, halakhah, commentaries, Kabbalah, Hasidut and responsa. Translate the passage the "
    "user gives into faithful, plain English. Translate everything, including quoted verses and "
    "abbreviations (expand them). Keep personal names and Hebrew technical terms transliterated "
    "where English has no common equivalent (e.g. Shabbat, eruv, tefillin). Add no commentary, "
    "notes or explanations. Output only the translation."
)


# Rabbinic abbreviations Qwen expanded wrongly in the D3a sample (הר"ב → "the Rambam"); added to
# the prompts from D3a block 2 on (2026-11-01).
GLOSSARY = (
    "\n\nAbbreviations of authorities and books (use these expansions when the abbreviation names "
    "a person or book; otherwise decide from context): "
    'הרע"ב = Rabbi Ovadiah of Bartenura (Bartenura); רש"י = Rashi; ר"ת = Rabbeinu Tam; '
    'ר"י in Tosafot and medieval commentaries = the Ri (Rabbi Yitzchak of Dampierre), in Mishnah '
    'and Talmud usually Rabbi Yehudah or Rabbi Yochanan; ר"ח as an authority = Rabbeinu Chananel, '
    'otherwise Rosh Chodesh; הרמב"ם = Rambam (Maimonides); הרמב"ן / רמב"ן = Ramban (Nachmanides); '
    'הרי"ף = the Rif; הרא"ש = the Rosh; הר"ן = the Ran; רשב"א = the Rashba; ריטב"א = the Ritva; '
    'רשב"ם = the Rashbam; ראב"ד = the Raavad; הרז"ה = Rabbi Zerachiah HaLevi (Baal HaMaor); '
    'נ"י = Nimukei Yosef; מהרש"א = the Maharsha; מהר"ם = the Maharam; רע"א = Rabbi Akiva Eiger; '
    'הגר"א = the Vilna Gaon (the Gra); תוס\' = Tosafot; תוי"ט / תי"ט = Tosafot Yom Tov; '
    'טור = the Tur; ב"י = Beit Yosef; שו"ע = Shulchan Arukh; רמ"א = the Rema; ש"ך = the Shakh; '
    'ט"ז = the Taz; מג"א = Magen Avraham; או"ח = Orach Chayim; יו"ד = Yoreh De\'ah; '
    'אה"ע = Even HaEzer; חו"מ = Choshen Mishpat; ש"ס = the Talmud; ת"ק = the first Tanna.'
)


# Hebrew-letter numerals: Qwen misreads tens letters (סי' קע"ג → 153, קפ"ח → 168, ח"י → 81) and a
# prompt note did not help (24-batch test, 2026-11-02), so from D3a block `numerals_from_block` on
# numerals after a reference word are replaced by their gematria value in the text sent to the model
# (2026-11-02). Only forms with geresh/gershayim after a marker, so words are left alone.
GEMATRIA = dict(
    zip(
        "אבגדהוזחטיכלמנסעפצקרשתךםןףץ",
        [*range(1, 10), *range(10, 100, 10), 100, 200, 300, 400, 20, 40, 50, 80, 90],
        strict=True,
    )
)
NUMERAL_RE = re.compile(
    r"(?<![א-ת])([בולהשמכ]{0,2}(סימנים|סימן|סי'|סעיף|סע'|ס\"ק|פרקים|פרק|הלכה|הל'|משנה|אות|"
    r"דף|עמוד|שורש|מצוה|ענף|דרוש|שער|פסוק|פס'|מזמור)\s)"
    r"([א-ת]{1,3}[\"״][א-ת]|[א-ת]['׳])(?![א-ת])"
)
# chapters, mishnayot and halakhot never reach 200, so after these ר"י, ר"ע, ר"פ… are rabbis or
# "beginning of chapter", not numbers
SMALL = {"פרק", "פרקים", "משנה", "הלכה", "הל'", "מזמור"}


def _gematria(m: re.Match[str]) -> str:
    letters = [c for c in m.group(3) if c in GEMATRIA]
    v = [GEMATRIA[c] for c in letters]
    # numerals are written largest letter first (ח"י = 18 is the usual exception) and repeat only
    # ת; anything else after a marker is an abbreviation (סי' הנ"ל = "the siman above", מ"מ)
    ordered = v == sorted(v, reverse=True) or m.group(3) in ('ח"י', "ח״י")
    repeated = any(letters.count(c) > 1 for c in set(letters) - {"ת"})
    if not ordered or repeated or (m.group(2) in SMALL and sum(v) > 150):
        return m.group(0)
    return m.group(1) + str(sum(v))


def gematria_digits(text: str) -> str:
    """סי' קע"ג → סי' 173, דף כ"ב ע"א → דף 22 ע"א; other Hebrew is left as it is."""
    return NUMERAL_RE.sub(_gematria, text)


# In commentaries on the Mishnah הר"ב is Bartenura; elsewhere (e.g. the Shakh) it is not, so this
# goes next to the text, only for those works.
MISHNAH_NOTE = '\nNote: in this work הר"ב means Rabbi Ovadiah of Bartenura (Bartenura).'


def _note(glossary: bool, categories: str) -> str:
    if glossary and categories.startswith("Mishnah >") and " on Mishnah" in categories:
        return MISHNAH_NOTE
    return ""


def messages(
    title: str,
    ref: str,
    text: str,
    glossary: bool = False,
    categories: str = "",
    numerals: bool = False,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM + (GLOSSARY if glossary else "")},
        {
            "role": "user",
            "content": f"Source: {ref} ({title}){_note(glossary, categories)}\n\n"
            + (gematria_digits(text) if numerals else text),
        },
    ]


def trial_sample(cfg: dict[str, Any]) -> pl.DataFrame:
    tc = load_yaml("configs/gen.yaml")["trial"]
    d1 = pl.read_parquet(resolve("data/train/d1_bitext.parquet")).filter(~pl.col("synthetic"))
    t = pl.read_parquet(
        resolve(cfg["processed"]["texts"]),
        columns=["ref", "lang", "title", "category", "categories", "text_clean", "n_words"],
    ).filter(pl.col("ref").is_in(d1["ref"].to_list()))
    he = t.filter(pl.col("lang") == "he").filter(
        pl.col("n_words").is_between(tc["min_words"], tc["max_words"])
        & ~pl.col("categories").str.starts_with("Tanakh > Torah")
        & ~pl.col("categories").str.starts_with("Tanakh > Prophets")
        & ~pl.col("categories").str.starts_with("Tanakh > Writings")
    )
    en = t.filter(pl.col("lang") == "en").select("ref", pl.col("text_clean").alias("reference"))
    parts = []
    for cat, n in tc["per_category"].items():
        x = he.filter(pl.col("category") == cat)
        # one segment per work first, so a few large works do not dominate
        x = x.sample(fraction=1.0, shuffle=True, seed=tc["seed"]).unique("title", keep="first")
        parts.append(x.head(n))
    s = pl.concat(parts).join(en, on="ref")
    return s.select("ref", "title", "category", pl.col("text_clean").alias("he"), "reference")


def run_trial(cfg: dict[str, Any]) -> None:
    gc = load_yaml("configs/gen.yaml")
    out = resolve(gc["trial"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    s = trial_sample(cfg)
    s.write_parquet(out / "sample.parquet")
    llm = LLM(gc["cache"], concurrency=gc["concurrency"], task="m3b_trial")
    for name, params in gc["settings"].items():
        reqs = [
            (messages(t, r, h), params) for r, t, h in s.select("ref", "title", "he").iter_rows()
        ]
        res = llm.map(reqs)
        rows = [
            {
                "ref": r,
                "translation": x.get("content"),
                "finish_reason": x.get("finish_reason"),
                "out_tokens": (x.get("usage") or {}).get("completion_tokens"),
                "seconds": x.get("seconds"),
                "error": x.get("error"),
            }
            for r, x in zip(s["ref"].to_list(), res, strict=True)
        ]
        pl.DataFrame(rows).write_parquet(out / f"{name}.parquet")
        errs = sum(r["error"] is not None for r in rows)
        log.info("trial %s: %d translations, %d errors", name, len(rows), errs)


def score_trial(model: str = "f2llm-14b") -> dict[str, Any]:
    """Teacher cosine of each translation with the human translation (and the Hebrew source);
    R@1 = the translation finds its own human translation among the 50."""
    from ein_mishpat_ai.eval.encoder import Encoder, model_spec

    gc = load_yaml("configs/gen.yaml")
    out = resolve(gc["trial"]["out_dir"])
    s = pl.read_parquet(out / "sample.parquet")
    enc = Encoder(model_spec(model, batch_size=16))
    ref = enc.encode(s["reference"].to_list())
    he = enc.encode(s["he"].to_list())
    res: dict[str, Any] = {
        "n": s.height,
        "human": {
            "cos_source": float(np.mean(np.sum(he * ref, 1))),
            "r1_source_to_text": float(np.mean(np.argmax(he @ ref.T, 1) == np.arange(s.height))),
        },
    }
    for name in gc["settings"]:
        t = pl.read_parquet(out / f"{name}.parquet")
        tr = enc.encode([x or "" for x in t["translation"].to_list()])
        res[name] = {
            "cos_human": float(np.mean(np.sum(tr * ref, 1))),
            "cos_source": float(np.mean(np.sum(tr * he, 1))),
            "r1_to_human": float(np.mean(np.argmax(tr @ ref.T, 1) == np.arange(s.height))),
            "r1_source_to_text": float(np.mean(np.argmax(he @ tr.T, 1) == np.arange(s.height))),
            "errors": int(t["error"].is_not_null().sum()),
            "truncated": int((t["finish_reason"] == "length").sum()),
            "out_tokens_mean": float(t["out_tokens"].mean() or 0),
            "seconds_mean": float(t["seconds"].mean() or 0),
        }
        t.with_columns(pl.Series("cos_human", np.sum(tr * ref, 1))).write_parquet(
            out / f"{name}.parquet"
        )
    (out / "scores.json").write_text(json.dumps(res, indent=2))
    log.info("trial scores: %s", res)
    return res


# --- batched requests: several consecutive segments of one work per request -----------------

SYSTEM_BATCH = SYSTEM.replace(
    "Translate the passage the user gives",
    "The user gives numbered segments of one work, in order (usually consecutive). Translate "
    "each segment",
).replace(
    "Output only the translation.",
    "Translate each segment on its own: never move, merge or skip text between segments; use "
    "the neighbouring segments only as context. Output only a JSON object mapping each segment "
    'number to its translation, e.g. {"1": "...", "2": "..."}, with exactly one entry per '
    "segment.",
)


def batch_messages(
    title: str,
    refs: list[str],
    texts: list[str],
    glossary: bool = False,
    categories: str = "",
    numerals: bool = False,
) -> list[dict[str, str]]:
    if numerals:
        texts = [gematria_digits(t) for t in texts]
    body = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts, 1))
    head = f"Source: {refs[0]} – {refs[-1]} ({title}){_note(glossary, categories)}"
    return [
        {"role": "system", "content": SYSTEM_BATCH + (GLOSSARY if glossary else "")},
        {"role": "user", "content": f"{head}\n\n{body}"},
    ]


def parse_batch(content: str | None, n: int) -> list[str] | None:
    """Translations in order, or None if the answer is not exactly n non-empty numbered items."""
    if not content:
        return None
    s = content.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        d = json.loads(s[s.index("{") : s.rindex("}") + 1])
    except ValueError:
        d = _loose_items(s)  # unescaped quotes inside a translation break strict JSON
    if not isinstance(d, dict) or sorted(d) != sorted(str(i) for i in range(1, n + 1)):
        return None
    out = [d[str(i)] for i in range(1, n + 1)]
    return out if all(isinstance(x, str) and x.strip() for x in out) else None


ITEM = re.compile(r'(?:^|,)\s*"(\d+)"\s*:\s*"', re.M)


def _loose_items(s: str) -> dict[str, str] | None:
    """Split `{"1": "...", "2": "..."}` at the item keys (each at a line start or after a comma)
    when strict JSON fails; numbers must run 1, 2, 3, … so a quoted '"3": "' inside a translation
    cannot split it."""
    body = s[s.find("{") + 1 : s.rfind("}")] if "{" in s and "}" in s else None
    if body is None:
        return None
    ms = list(ITEM.finditer(body))
    keep = []
    for m in ms:
        if int(m.group(1)) == len(keep) + 1:
            keep.append(m)
    if not keep:
        return None
    out = {}
    for m, nxt in zip(keep, [*keep[1:], None], strict=True):
        v = body[m.end() : nxt.start() if nxt else len(body)].rstrip().rstrip(",").rstrip()
        if not v.endswith('"'):
            return None
        out[m.group(1)] = v[:-1].replace('\\"', '"').replace("\\n", "\n")
    return out


def batch_sample(cfg: dict[str, Any]) -> pl.DataFrame:
    """Runs of `size` consecutive segments (same node, consecutive seg_order) that all have a
    free human translation, one run per work, stratified by genre."""
    bc = load_yaml("configs/gen.yaml")["batch_test"]
    size = bc["size"]
    d1 = pl.read_parquet(resolve("data/train/d1_bitext.parquet")).filter(~pl.col("synthetic"))
    t = pl.read_parquet(
        resolve(cfg["processed"]["texts"]),
        columns=[
            "ref",
            "lang",
            "title",
            "category",
            "categories",
            "node_ref",
            "seg_order",
            "text_clean",
            "n_words",
        ],
    ).filter(pl.col("ref").is_in(d1["ref"].implode()))
    he = t.filter(
        (pl.col("lang") == "he")
        & pl.col("n_words").is_between(bc["min_words"], bc["max_words"])
        & ~pl.col("categories").str.contains(r"^Tanakh > (Torah|Prophets|Writings)")
    ).sort("node_ref", "seg_order")
    # run id: consecutive seg_order within a node
    he = he.with_columns(
        (
            (pl.col("seg_order").diff().over("node_ref").fill_null(1) != 1)
            .cum_sum()
            .over("node_ref")
        ).alias("_run")
    )
    runs = (
        he.group_by("node_ref", "_run")
        .agg(pl.len().alias("n"), pl.first("title"), pl.first("category"))
        .filter(pl.col("n") >= size)
    )
    en = t.filter(pl.col("lang") == "en").select("ref", pl.col("text_clean").alias("reference"))
    rows = []
    for cat, k in bc["per_category"].items():
        r = runs.filter(pl.col("category") == cat).sample(
            fraction=1.0, shuffle=True, seed=bc["seed"]
        )
        r = r.unique("title", keep="first").head(k)
        for b, (node, run) in enumerate(r.select("node_ref", "_run").iter_rows()):
            seg = he.filter((pl.col("node_ref") == node) & (pl.col("_run") == run)).head(size)
            rows.append(seg.with_columns(pl.lit(f"{cat}-{b}").alias("batch")))
    s = pl.concat(rows).join(en, on="ref").sort("batch", "seg_order")
    return s.select(
        "batch", "ref", "title", "category", pl.col("text_clean").alias("he"), "reference"
    )


def run_batch_test(cfg: dict[str, Any]) -> None:
    import time

    gc = load_yaml("configs/gen.yaml")
    bc = gc["batch_test"]
    out = resolve(bc["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    s = batch_sample(cfg)
    s.write_parquet(out / "sample.parquet")
    llm = LLM(gc["cache"], concurrency=gc["concurrency"], task="m3b_batch_test")
    single = gc["settings"]["no_think"]
    t0 = time.time()
    res = llm.map(
        [(messages(t, r, h), single) for r, t, h in s.select("ref", "title", "he").iter_rows()]
    )
    t_single = time.time() - t0
    s = s.with_columns(
        pl.Series("single", [x.get("content") for x in res]),
        pl.Series(
            "single_out_tokens", [(x.get("usage") or {}).get("completion_tokens") for x in res]
        ),
        pl.Series("single_in_tokens", [(x.get("usage") or {}).get("prompt_tokens") for x in res]),
    )
    groups = s.group_by("batch", maintain_order=True).agg("ref", "he", pl.first("title"))
    params = {**single, "max_tokens": bc["max_tokens"]}
    t0 = time.time()
    bres = llm.map([(batch_messages(t, r, h), params) for _, r, h, t in groups.iter_rows()])
    t_batch = time.time() - t0
    trans: dict[str, str | None] = {}
    meta = []
    for (b, refs, _, _), x in zip(groups.iter_rows(), bres, strict=True):
        parsed = parse_batch(x.get("content"), len(refs))
        meta.append(
            {
                "batch": b,
                "parsed": parsed is not None,
                "finish_reason": x.get("finish_reason"),
                "in_tokens": (x.get("usage") or {}).get("prompt_tokens"),
                "out_tokens": (x.get("usage") or {}).get("completion_tokens"),
                "seconds": x.get("seconds"),
                "raw": x.get("content"),
            }
        )
        for i, r in enumerate(refs):
            trans[r] = parsed[i] if parsed else None
    s = s.with_columns(pl.Series("batched", [trans[r] for r in s["ref"].to_list()]))
    s.write_parquet(out / "translations.parquet")
    pl.DataFrame(meta).write_parquet(out / "batches.parquet")
    (out / "timing.json").write_text(
        json.dumps({"wall_single": t_single, "wall_batched": t_batch}, indent=2)
    )
    log.info(
        "batch test: %d segments, %d batches parsed of %d",
        s.height,
        sum(m["parsed"] for m in meta),
        len(meta),
    )


def score_batch_test(model: str = "f2llm-14b") -> dict[str, Any]:
    """Teacher check per batch: each translation must match its own source best among the batch
    (same check for single-segment translations, as a baseline)."""
    from ein_mishpat_ai.eval.encoder import Encoder, model_spec

    bc = load_yaml("configs/gen.yaml")["batch_test"]
    out = resolve(bc["out_dir"])
    s = pl.read_parquet(out / "translations.parquet")
    m = pl.read_parquet(out / "batches.parquet")
    enc = Encoder(model_spec(model, batch_size=16))
    he = enc.encode(s["he"].to_list())
    ref = enc.encode(s["reference"].to_list())
    single = enc.encode([x or "" for x in s["single"].to_list()])
    ok = s["batched"].is_not_null().to_numpy()
    batched = enc.encode([x or "" for x in s["batched"].to_list()])
    b = s["batch"].to_numpy()
    res: dict[str, Any] = {
        "segments": s.height,
        "batches": m.height,
        "batches_parsed": int(m["parsed"].sum()),
    }
    cols = {}
    for name, tr in (("single", single), ("batched", batched)):
        own = np.zeros(s.height, bool)
        for g in np.unique(b):
            idx = np.where(b == g)[0]
            sim = he[idx] @ tr[idx].T
            own[idx] = sim.argmax(1) == np.arange(len(idx))
        cos_src = np.sum(he * tr, 1)
        cos_hum = np.sum(ref * tr, 1)
        cols[name] = (own, cos_src, cos_hum)
        res[name] = {
            "cos_source": float(cos_src[ok].mean()),
            "cos_human": float(cos_hum[ok].mean()),
            "own_source_best_in_batch": float(own[ok].mean()),
        }
    batch_ok = {g: bool(cols["batched"][0][b == g].all()) for g in np.unique(b) if ok[b == g].all()}
    res["batches_pass_teacher_check"] = int(sum(batch_ok.values()))
    res["cos_batched_vs_single"] = float(np.sum(batched * single, 1)[ok].mean())
    res["human_cos_source"] = float(np.sum(he * ref, 1).mean())
    res["tokens"] = {
        "single_in": int(s["single_in_tokens"].sum()),
        "single_out": int(s["single_out_tokens"].sum()),
        "batched_in": int(m["in_tokens"].sum()),
        "batched_out": int(m["out_tokens"].sum()),
        "single_s_per_request": None,
        "batched_s_per_request": float(m["seconds"].mean()),
    }
    res["timing"] = json.loads((out / "timing.json").read_text())
    s.with_columns(
        pl.Series("cos_src_single", cols["single"][1]),
        pl.Series("cos_src_batched", cols["batched"][1]),
        pl.Series("own_best_batched", cols["batched"][0]),
    ).write_parquet(out / "translations.parquet")
    (out / "scores.json").write_text(json.dumps(res, indent=2))
    log.info("batch test scores: %s", res)
    return res
