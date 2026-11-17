"""KesherBench with the Masoretic text of the Torah (2026-11-12).

The version rule of data/pairs.py chose "Miqra Mevoar" (a modern-Hebrew rendering) as the Hebrew of the Torah and
the Five Megillot. This script builds a second copy of the benchmark in which those passages carry the Masoretic text
("Tanach with Nikkud", vowel points removed, the same version the rest of the Tanakh uses), and re-runs the settings
that search them, with the unchanged harness (src/ is not touched; only the benchmark text differs):

  K2a commentary → Torah verse, he→he and en→he   (corpus torah_he: all 5,846 verses change)
  K5 topic → sources, he→he and en→he            (pool k5pool_he: 4,077 of 638,175 passages change)

usage: python scripts/masoretic.py build              → data/bench/kesher_mt (other files are links to data/bench/kesher)
       python scripts/masoretic.py run MODEL [--k5] [--demo]   (GPU; one model load for encoding and both splits)
       python scripts/masoretic.py bm25               (CPU)
       python scripts/masoretic.py train-mix          (CPU; the training mix with the Masoretic text, for the training test)
       python scripts/masoretic.py scale-rule         (CPU; the 14B adapter-scale rule on the corrected dev results)
       python scripts/masoretic.py apply              (CPU; the two Masoretic corpora become data/bench/kesher's)
       python scripts/masoretic.py install            (CPU; apply + this project's saved vectors and results)
       python scripts/masoretic.py demo               (CPU; the demo index with the Masoretic rows)
       python scripts/masoretic.py voc MODEL          (GPU; K2a with vocalised Masoretic text, for the vowel-point report)
Config of the runs: scripts/jobs/kesher_masoretic.yaml = configs/eval/kesher.yaml with the three folders changed.
K5 needs the model's saved pool vectors (artifacts/emb/kesher/MODEL/k5pool_he): only the changed rows are encoded.
--demo: also encode the demo's Torah and Megillot passages (data/bench/kesher_mt/changed/demo.parquet).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from ein_mishpat_ai.utils.io import load_yaml, resolve

OLD, NEW = resolve("data/bench/kesher"), resolve("data/bench/kesher_mt")
CFG = "scripts/jobs/kesher_masoretic.yaml"
KEPT = resolve("data/bench/_superseded/kesher_miqra_mevoar")  # the two corpora as the pipeline wrote them
EMB_KEPT = resolve("artifacts/emb/_superseded/kesher_miqra_mevoar")
EVAL_KEPT = resolve("artifacts/eval/_superseded/kesher_miqra_mevoar_20261112")
VERSION = "Tanach with Nikkud"
TITLES = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Ecclesiastes", "Esther", "Lamentations",
          "Song of Songs", "Ruth"]
K2A = ["K2a_commentary_torah/he2he", "K2a_commentary_torah/en2he"]
K5 = ["K5_topic_sources/he2he", "K5_topic_sources/en2he"]


def masoretic() -> pl.DataFrame:
    """ref, text (text_plain of the Masoretic version), license for the Torah and the Five Megillot."""
    v = pl.read_parquet(resolve("data/meta/versions.parquet"))
    ids = v.filter((pl.col("version_title") == VERSION) & pl.col("title").is_in(TITLES) & (pl.col("lang") == "he"))
    assert ids.height == len(TITLES), ids
    seg = pl.scan_parquet(resolve("data/processed/segments.parquet"))
    m = (seg.filter(pl.col("version_id").is_in(ids["version_id"].to_list()))
         .select(pl.col("ref").alias("id"), pl.col("text_plain").alias("text_mt"), pl.col("license").alias("license_mt"))
         .collect())
    assert m["id"].is_unique().all() and m["text_mt"].null_count() == 0
    return m


def vocalised() -> dict[str, str]:
    """ref -> the same Masoretic text with its vowel points (text_clean), for the vowel-point checks."""
    v = pl.read_parquet(resolve("data/meta/versions.parquet"))
    vid = v.filter((pl.col("version_title") == VERSION) & pl.col("title").is_in(TITLES) & (pl.col("lang") == "he"))
    seg = (pl.scan_parquet(resolve("data/processed/segments.parquet"))
           .filter(pl.col("version_id").is_in(vid["version_id"].to_list())).select("ref", "text_clean").collect())
    return dict(zip(seg["ref"].to_list(), seg["text_clean"].to_list(), strict=True))


def build() -> None:
    m = masoretic()
    if NEW.exists():
        shutil.rmtree(NEW)
    for f in sorted(OLD.rglob("*")):  # everything else is the same file
        rel = f.relative_to(OLD)
        if f.is_dir():
            (NEW / rel).mkdir(parents=True, exist_ok=True)
        elif rel.as_posix() not in ("corpora/torah_he.parquet", "corpora/k5pool_he.parquet"):
            (NEW / rel).parent.mkdir(parents=True, exist_ok=True)
            (NEW / rel).symlink_to(f)
    (NEW / "changed").mkdir(exist_ok=True)
    stats = {}
    for name in ("torah_he", "k5pool_he"):
        orig = KEPT / "corpora" / f"{name}.parquet"  # after `install`, the pipeline's own files are kept here
        c = pl.read_parquet(orig if orig.exists() else OLD / "corpora" / f"{name}.parquet")
        j = c.join(m, on="id", how="left")
        hit = j["text_mt"].is_not_null()
        # only passages that carried the rendering are replaced
        t = pl.read_parquet(resolve("data/processed/texts.parquet"), columns=["ref", "lang", "version_title"])
        mev = set(t.filter((pl.col("lang") == "he") & pl.col("version_title").str.contains("Miqra Mevoar"))["ref"])
        is_mev = j["id"].is_in(list(mev))
        assert (hit | ~is_mev).all(), f"{name}: a Miqra Mevoar passage has no Masoretic text"
        rep = hit & is_mev
        out = j.with_columns(
            pl.when(rep).then(pl.col("text_mt")).otherwise(pl.col("text")).alias("text"),
            pl.when(rep).then(pl.col("license_mt")).otherwise(pl.col("license")).alias("license"),
        ).select(c.columns)
        assert out["id"].to_list() == c["id"].to_list()
        out.write_parquet(NEW / "corpora" / f"{name}.parquet", compression="zstd")
        out.filter(rep).select("id", "text").write_parquet(NEW / "changed" / f"{name}.parquet")
        stats[name] = {"rows": c.height, "changed": int(rep.sum())}
    assert stats["torah_he"]["changed"] == stats["torah_he"]["rows"]
    # demo: every Torah / Megillot passage of the demo corpus, Masoretic text, the demo's 50-character rule
    demo = pl.read_parquet(resolve("release/space/index/passages.parquet"), columns=["id", "title"])
    meta = load_yaml("configs/data.yaml")
    del meta
    d = m.filter(pl.col("text_mt").str.len_chars() >= 50).select("id", pl.col("text_mt").alias("text"))
    d.write_parquet(NEW / "changed" / "demo.parquet")
    stats["demo"] = {"rows_now_from_these_books": demo.filter(pl.col("title").is_in(TITLES)).height, "masoretic_ge_50": d.height}
    cfg = load_yaml("configs/eval/kesher.yaml")
    cfg.update(bench_dir="data/bench/kesher_mt", emb_dir="artifacts/emb/kesher_mt", out_dir="artifacts/eval/kesher_mt")
    resolve(CFG).write_text("# KesherBench with the Masoretic Torah text (scripts/masoretic.py): configs/eval/kesher.yaml "
                            "with the three folders changed.\n" + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=1000))
    (NEW / "MASORETIC.json").write_text(__import__("json").dumps({"version": VERSION, "titles": TITLES, **stats}, indent=1) + "\n")
    print(stats)


def _cache_encoder() -> None:
    """One model load per process: encoding and both splits ask for the same spec."""
    import ein_mishpat_ai.eval.encode as enc_mod
    import ein_mishpat_ai.eval.encoder as encoder_mod

    orig, cache = encoder_mod.Encoder, {}

    def cached(spec, device: str = "cuda"):  # noqa: ANN001, ANN202
        key = repr(spec)
        if key not in cache:
            cache[key] = orig(spec, device)
        return cache[key]

    encoder_mod.Encoder = cached
    enc_mod.Encoder = cached


def run(model: str, k5: bool, demo: bool) -> None:
    from ein_mishpat_ai.eval.encode import encode_table, load_embeddings
    from ein_mishpat_ai.eval.kesher import evaluate

    _cache_encoder()
    cfg = load_yaml(CFG)
    emb = resolve(cfg["emb_dir"]) / model
    dp = load_yaml("configs/eval/models.yaml")["models"][model].get("doc_prompt", "")
    kw = dict(prefix=dp, max_length=cfg["max_length"], batch_size=cfg["batch_size"])
    encode_table(model, NEW / "corpora/torah_he.parquet", emb / "torah_he", **kw)
    only = list(K2A)
    if k5:
        encode_table(model, NEW / "changed/k5pool_he.parquet", emb / "k5pool_changed", **kw)
        dst = emb / "k5pool_he"
        if not (dst / "shard00_part0000.npy").exists():
            ids, mat = load_embeddings(resolve("artifacts/emb/kesher") / model / "k5pool_he")
            cid, cmat = load_embeddings(emb / "k5pool_changed")
            row = {x: i for i, x in enumerate(ids)}
            assert len(ids) == pl.read_parquet(NEW / "corpora/k5pool_he.parquet", columns=["id"]).height
            mat = np.array(mat, dtype=np.float16)
            mat[[row[x] for x in cid]] = cmat
            dst.mkdir(parents=True, exist_ok=True)
            pl.DataFrame({"id": ids}).write_parquet(dst / "shard00_part0000.parquet")
            np.save(dst / "shard00_part0000.tmp.npy", mat)
            (dst / "shard00_part0000.tmp.npy").replace(dst / "shard00_part0000.npy")
            del mat
        only += K5
    if demo:
        encode_table(model, NEW / "changed/demo.parquet", resolve("artifacts/emb/demo_mt") / model, **kw)
    for split in ("test", "dev"):
        rec = evaluate(model, CFG, split, only)
        print(model, split, {k: round(v, 4) for k, v in rec["metrics"].items() if k.endswith(("ndcg_at_10", "hit_at_10"))},
              flush=True)


def voc(model: str) -> None:
    """Vowel-point check for K2a with the Masoretic text, as scripts/followup_checks.py does it for the other
    settings: the corpus with its vowel points (text_clean of the same version), Hebrew queries vocalised where
    their vocalised text exists. → artifacts/eval/niqqud_mt/MODEL/results.json"""
    import json

    from ein_mishpat_ai.data.normalize import plain_hebrew
    from ein_mishpat_ai.eval.encoder import Encoder, model_spec
    from ein_mishpat_ai.eval.metrics import qrels_metrics, topk
    from ein_mishpat_ai.utils.code_hash import code_hash
    from ein_mishpat_ai.utils.io import now_iso

    out = resolve(f"artifacts/eval/niqqud_mt/{model}")
    if (out / "results.json").exists():
        return
    kc = load_yaml(CFG)
    cvoc = vocalised()
    corpus = pl.read_parquet(NEW / "corpora/torah_he.parquet", columns=["id", "text"])
    ids = corpus["id"].to_list()
    for i, t in zip(ids, corpus["text"].to_list(), strict=True):  # the same text except for the marks
        assert plain_hebrew(cvoc[i]) == t, i
    texts = pl.read_parquet(resolve(load_yaml("configs/data.yaml")["processed"]["texts"]),
                            columns=["ref", "lang", "text_clean"]).filter(pl.col("lang") == "he")
    qvoc = dict(zip(texts["ref"].to_list(), texts["text_clean"].to_list(), strict=True))
    marks = __import__("re").compile("[\u05B0-\u05BD\u05BF\u05C1\u05C2\u05C7]")
    harness = json.loads((resolve(kc["out_dir"]) / "test" / model / "results.json").read_text())["metrics"]
    enc = Encoder(model_spec(model, max_length=kc["max_length"], batch_size=kc["batch_size"]))
    emb = enc.encode([cvoc[i] for i in ids]).astype(np.float16).astype(np.float32)  # stored as float16, like the harness
    col = {c: i for i, c in enumerate(ids)}
    rec: dict = {"model": model, "kesher": {}}
    task = "K2a_commentary_torah"
    for setting in ("he2he", "en2he"):
        q = pl.read_parquet(NEW / task / f"queries_{setting}_test.parquet")
        pos: dict[str, set[int]] = {}
        for qid, cid in pl.read_parquet(NEW / task / "qrels_test.parquet").select("query_id", "corpus_id").iter_rows():
            if cid in col:
                pos.setdefault(qid, set()).add(col[cid])
        q = q.filter(pl.col("id").is_in(list(pos)))
        qt = [qvoc.get(i, t) for i, t in zip(q["id"].to_list(), q["text"].to_list(), strict=True)] \
            if setting.startswith("he") else q["text"].to_list()
        qe = enc.encode(qt, prefix=enc.query_prefix(kc["tasks"][task]))
        _, idx = topk(qe, emb, max(kc["k_values"]))
        mm = qrels_metrics(idx, [pos[i] for i in q["id"].to_list()], kc["k_values"], kc["mrr_k"])
        ts = f"{task}/{setting}"
        rec["kesher"][ts] = {"n": q.height, "query_vocalised_share": float(np.mean([bool(marks.search(t)) for t in qt])),
                             "corpus_vocalised_share": 1.0,
                             "plain_ndcg10": harness[f"{ts}/ndcg_at_10"], "vocalised_ndcg10": mm["ndcg_at_10"],
                             "plain_r100": harness[f"{ts}/recall_at_100"], "vocalised_r100": mm["recall_at_100"]}
    rec.update(code_hash=code_hash(), finished_at=now_iso())
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rec, indent=2))
    print(model, {k: (round(x["plain_ndcg10"], 3), round(x["vocalised_ndcg10"], 3)) for k, x in rec["kesher"].items()}, flush=True)


def apply() -> None:
    """Put the two Masoretic corpora into data/bench/kesher; the pipeline's own files are kept beside it."""
    for name in ("torah_he", "k5pool_he"):
        cur, kept = OLD / "corpora" / f"{name}.parquet", KEPT / "corpora" / f"{name}.parquet"
        if not kept.exists():
            kept.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(cur, kept)
        shutil.copyfile(NEW / "corpora" / f"{name}.parquet", cur)


def install() -> None:
    """Make the Masoretic text the benchmark: corpora, corpus vectors and results move into the usual folders
    (data/bench/kesher, artifacts/emb/kesher, artifacts/eval/kesher), so every report and the release are built from
    them without a config change. What was there before is kept under _superseded. Safe to run again: it only adds
    what is new.

    Results: for every model and split run by `run`, the metrics and saved top lists of the re-run settings replace the
    old ones; the other settings do not search the Torah and stay. A model whose topic-search pool could not be
    re-run (its pool vectors were deleted earlier) keeps its K5 numbers from the earlier pool; MASORETIC.json in the
    results folder lists which is which."""
    import json

    cfg, old_cfg = load_yaml(CFG), load_yaml("configs/eval/kesher.yaml")
    apply()  # 1. benchmark files
    # 2. corpus vectors
    new_emb, emb = resolve(cfg["emb_dir"]), resolve(old_cfg["emb_dir"])
    for m in sorted(x for x in new_emb.iterdir() if x.is_dir()):
        for cname in ("torah_he", "k5pool_he"):
            src, dst = m / cname, emb / m.name / cname
            if not (src / "shard00_part0000.npy").exists():
                continue
            if dst.exists():
                keep = EMB_KEPT / m.name / cname
                if keep.exists():
                    shutil.rmtree(dst)  # already kept once: this is an earlier install of the new vectors
                else:
                    keep.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(dst, keep)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dst)
    # 3. results
    new_ev, ev = resolve(cfg["out_dir"]), resolve(old_cfg["out_dir"])
    if not EVAL_KEPT.exists():
        shutil.copytree(ev, EVAL_KEPT)
    manifest: dict = {"text": VERSION, "books": TITLES, "installed": __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).isoformat(timespec="seconds"), "models": {}}
    for split in ("test", "dev"):
        for d in sorted(x for x in (new_ev / split).iterdir() if (x / "results.json").exists()):
            tgt = ev / split / d.name
            if not (tgt / "results.json").exists():
                continue
            nr = json.loads((d / "results.json").read_text())
            rec = json.loads((EVAL_KEPT / split / d.name / "results.json").read_text())
            done = sorted({"/".join(k.split("/")[:2]) for k in nr["metrics"]})
            for k in [k for k in rec["metrics"] if "/".join(k.split("/")[:2]) in done]:
                del rec["metrics"][k]
            rec["metrics"].update(nr["metrics"])
            assert all(rec["n_queries"][ts] == n for ts, n in nr["n_queries"].items()), (d.name, split)
            rec["masoretic"] = {"settings": done, "code_hash": nr["code_hash"], "finished_at": nr["finished_at"],
                                "k5_pool": "masoretic" if K5[0] in done or d.name == "bm25" else "earlier rendering"}
            (tgt / "results.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
            for f in sorted((d / "topk").glob("*.npz")) if (d / "topk").exists() else []:
                shutil.copyfile(f, tgt / "topk" / f.name)
            manifest["models"].setdefault(d.name, {})[split] = rec["masoretic"]
    untouched = sorted(x.name for sp in ("test", "dev") for x in (ev / sp).iterdir()
                       if x.is_dir() and x.name not in manifest["models"])
    manifest["not_rerun"] = sorted(set(untouched))
    (ev / "MASORETIC.json").write_text(json.dumps(manifest, indent=1) + "\n")
    # 4. vowel-point check: the K2a rows
    for d in sorted(x for x in resolve("artifacts/eval/niqqud_mt").glob("*") if (x / "results.json").exists()):
        f = resolve(f"artifacts/eval/niqqud/{d.name}/results.json")
        if not f.exists():
            continue
        keep = f.with_name("results.miqra_mevoar.json")
        if not keep.exists():
            shutil.copyfile(f, keep)
        rec = json.loads(keep.read_text())
        rec["kesher"].update(json.loads((d / "results.json").read_text())["kesher"])
        rec["masoretic_k2a"] = True
        f.write_text(json.dumps(rec, indent=2))
    print(json.dumps({m: {sp: v["k5_pool"] for sp, v in x.items()} for m, x in manifest["models"].items()}, indent=1))
    print("not re-run (results as before):", manifest["not_rerun"])


def demo() -> None:
    """The demo index with the Masoretic text for the Torah and the Five Megillot (their old rows are dropped, the
    passages of at least 50 characters come from the vectors `run --demo` wrote) and with every verse of the Tanakh,
    however short (2026-11-12: the 50-character rule stays for the other works; vectors of the short verses in
    artifacts/emb/demo_short). The index as it first was is kept."""
    import json

    from ein_mishpat_ai.eval.encode import load_embeddings
    from ein_mishpat_ai.utils.code_hash import code_hash
    from ein_mishpat_ai.utils.io import now_iso

    D = resolve("release/space/index")
    meta = json.loads((D / "meta.json").read_text())
    keep = resolve(".tmp/demo_index_miqra_mevoar_20261112")
    if not keep.exists():
        shutil.copytree(D, keep)
    P = pl.read_parquet(keep / "passages.parquet")
    E = np.load(keep / "embeddings.npy")
    ids, emb = load_embeddings(resolve("artifacts/emb/demo_mt") / meta["local_model"])
    new = pl.read_parquet(NEW / "changed/demo.parquet")
    assert sorted(ids) == sorted(new["id"].to_list())
    text = dict(zip(new["id"].to_list(), new["text"].to_list(), strict=True))
    t = pl.read_parquet(resolve("data/processed/texts.parquet"),
                        columns=["ref", "lang", "title", "license", "synthetic", "text_clean"])
    en = t.filter((pl.col("lang") == "en") & ~pl.col("synthetic") & (pl.col("license") != "unknown"))
    en_of = {r: (x, lic) for r, x, lic in en.select("ref", "text_clean", "license").iter_rows()}
    title = dict(t.filter(pl.col("lang") == "he").select("ref", "title").iter_rows())
    rows = pl.DataFrame({"id": ids, "title": [title[i] for i in ids], "work": ["Tanakh"] * len(ids),
                         "he": [text[i] for i in ids], "en": [en_of.get(i, (None, None))[0] for i in ids],
                         "license_he": ["public_domain"] * len(ids),
                         "license_en": [en_of.get(i, (None, None))[1] for i in ids]}, schema=P.schema)
    stay = ~P["title"].is_in(TITLES).to_numpy()
    # every Tanakh verse under 50 characters (Masoretic text for the Torah and the Megillot)
    sids, semb = load_embeddings(resolve("artifacts/emb/demo_short") / meta["local_model"])
    short = pl.read_parquet(NEW / "changed/demo_short_tanakh.parquet")
    assert sorted(sids) == sorted(short["id"].to_list()) and not set(sids) & (set(ids) | set(P.filter(pl.Series(stay))["id"]))
    stext = dict(zip(short["id"].to_list(), short["text"].to_list(), strict=True))
    lic = dict(t.filter(pl.col("lang") == "he").select("ref", "license").iter_rows())
    mt_ids = set(masoretic()["id"].to_list())
    srows = pl.DataFrame({"id": sids, "title": [title[i] for i in sids], "work": ["Tanakh"] * len(sids),
                          "he": [stext[i] for i in sids], "en": [en_of.get(i, (None, None))[0] for i in sids],
                          "license_he": ["public_domain" if i in mt_ids else lic[i] for i in sids],
                          "license_en": [en_of.get(i, (None, None))[1] for i in sids]}, schema=P.schema)
    passages = pl.concat([P.filter(pl.Series(stay)), rows, srows])
    vec = np.concatenate([E[stay], emb.astype(np.float16), semb.astype(np.float16)])
    assert passages.height == vec.shape[0] and passages["id"].is_unique().all() and passages["he"].null_count() == 0
    assert np.allclose(np.linalg.norm(vec[-len(ids):].astype(np.float32), axis=1), 1, atol=1e-2)
    np.save(D / "embeddings.npy", vec)
    passages.write_parquet(D / "passages.parquet", compression="zstd")
    meta.update(rows=int(vec.shape[0]), built=now_iso(), code_hash=code_hash(),
                torah_text="Masoretic (Tanach with Nikkud, vowel points removed)", min_chars_applies_to="all works except the Tanakh")
    (D / "meta.json").write_text(json.dumps(meta, indent=2))
    print("demo index:", P.height, "->", passages.height, "passages;", int((~stay).sum()), "old rows replaced by", len(ids),
          "+", len(sids), "short Tanakh verses")


def train_mix() -> None:
    """The training mix with the Masoretic text for the Torah and the Five Megillot: data/train/mix_d3b_all_mt.
    Same examples (the file is a link: same pairs, negatives, teacher scores, order), same texts except the Hebrew of
    the passages that carried the rendering. For the test run that asks whether the text matters in training
    (configs/train/abl_1.7b_p20_mt.yaml against student-1.7b-p20-d3b). The teacher scores of the changed passages
    were computed on the rendering and are not recomputed: only the text differs between the two runs."""
    import json

    src, dst = resolve("data/train/mix_d3b_all"), resolve("data/train/mix_d3b_all_mt")
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("examples.parquet", "dev_d1.parquet"):
        if not (dst / f).exists():
            (dst / f).symlink_to(src / f)
    m = masoretic().rename({"id": "ref"})
    t = pl.read_parquet(resolve("data/processed/texts.parquet"), columns=["ref", "lang", "version_title", "text_plain"])
    mev = t.filter((pl.col("lang") == "he") & pl.col("version_title").str.contains("Miqra Mevoar")).select("ref", "text_plain")
    tx = pl.read_parquet(src / "texts.parquet")
    j = (tx.join(m.select("ref", "text_mt"), on="ref", how="left")
         .join(mev.rename({"text_plain": "text_mev"}), on="ref", how="left"))
    rep_ = (j["lang"] == "he") & j["text_mev"].is_not_null()
    assert (j.filter(rep_)["text"] == j.filter(rep_)["text_mev"]).all(), "a passage to replace does not carry the rendering"
    assert j.filter(rep_)["text_mt"].null_count() == 0
    out = j.with_columns(pl.when(rep_).then(pl.col("text_mt")).otherwise(pl.col("text")).alias("text")).select(tx.columns)
    assert out.select("lang", "ref").equals(tx.select("lang", "ref"))
    out.write_parquet(dst / "texts.parquet")
    st = json.loads((src / "stats.json").read_text())
    st["masoretic"] = {"version": VERSION, "texts_replaced": int(rep_.sum()), "of_texts": tx.height,
                       "note": "examples.parquet and dev_d1.parquet are links to mix_d3b_all"}
    (dst / "stats.json").write_text(json.dumps(st, indent=1) + "\n")
    (dst / ".built").touch()
    print("mix_d3b_all_mt:", int(rep_.sum()), "of", tx.height, "texts replaced")


def scale_rule() -> None:
    """The rule of scripts/lora_scale.py (fixed 2026-11-09, before any scaled model was evaluated), applied to the dev
    results on the benchmark with the Masoretic text: eligible scales keep the KesherBench dev score within 0.01 of
    the flagship (s = 1); among them the highest topic hit@10 (mean of Hebrew and English) wins, and it replaces the
    flagship only if that is higher than the flagship's. Candidates: s = 1, 0.85, 0.7. s = 0.5 is not run again: on
    every earlier measurement it loses more of the score than 0.7 does. Writes
    artifacts/eval/lora_scale/choice_masoretic.json and prints the chosen model; the release is not changed by it."""
    import json

    sys.path.insert(0, str(resolve("scripts")))
    from lora_scale import FLAG, TOL, name

    from ein_mishpat_ai.eval.kesher import kesher_score, topic_hit10
    from ein_mishpat_ai.utils.io import now_iso

    ev = resolve("artifacts/eval/kesher/dev")
    rows = []
    for s_, mname in ((1.0, FLAG), (0.85, name("85")), (0.7, name("70"))):
        r = json.loads((ev / mname / "results.json").read_text())
        if r.get("masoretic", {}).get("k5_pool") == "earlier rendering":
            raise SystemExit(f"{mname}: its dev results are not on the Masoretic benchmark yet")
        if mname == name("85") and "masoretic" not in r and r["finished_at"] < "2026-11-12":
            raise SystemExit(f"{mname}: its dev results are from before the Torah-text change; evaluate it first")
        rows.append((s_, mname, r["metrics"]))
    base = rows[0][2]
    eligible = [x for x in rows if kesher_score(x[2]) >= kesher_score(base) - TOL]
    best = max(eligible, key=lambda x: (topic_hit10(x[2]), x[0]))
    pick = best[1] if topic_hit10(best[2]) > topic_hit10(base) else FLAG
    rec = {"model": pick, "rule": scale_rule.__doc__.split("Writes")[0].strip(),
           "dev": {m_: {"scale": s_, "score": kesher_score(r), "topic_hit10": topic_hit10(r),
                        "eligible": kesher_score(r) >= kesher_score(base) - TOL} for s_, m_, r in rows},
           "chosen_at": now_iso()}
    out = resolve("artifacts/eval/lora_scale/choice_masoretic.json")
    out.write_text(json.dumps(rec, indent=2) + "\n")
    for m_, d in rec["dev"].items():
        print(f"{m_}: s={d['scale']} dev score {d['score']:.4f} topic {d['topic_hit10']:.4f} eligible={d['eligible']}", file=sys.stderr)
    print(pick)


def bm25() -> None:
    from ein_mishpat_ai.eval.kesher import evaluate

    for split in ("test", "dev"):
        rec = evaluate("bm25", CFG, split, [K2A[0], K5[0]])
        print("bm25", split, {k: round(v, 4) for k, v in rec["metrics"].items() if k.endswith(("ndcg_at_10", "hit_at_10"))})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("bm25")
    sub.add_parser("train-mix")
    sub.add_parser("scale-rule")
    sub.add_parser("apply")
    sub.add_parser("install")
    sub.add_parser("demo")
    vv = sub.add_parser("voc")
    vv.add_argument("model")
    r = sub.add_parser("run")
    r.add_argument("model")
    r.add_argument("--k5", action="store_true")
    r.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "bm25":
        bm25()
    elif a.cmd == "train-mix":
        train_mix()
    elif a.cmd == "scale-rule":
        scale_rule()
    elif a.cmd == "apply":
        apply()
    elif a.cmd == "install":
        install()
    elif a.cmd == "demo":
        demo()
    elif a.cmd == "voc":
        voc(a.model)
    else:
        run(a.model, a.k5, a.demo)
    _ = Path
