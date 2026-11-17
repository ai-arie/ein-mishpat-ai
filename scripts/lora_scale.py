"""Flagship with its LoRA adapter scaled down before merging (follow-up check, 2026-11-09).

Idea: the adapter is a learned change ΔW added to the frozen base F2LLM-v2-14B. Merging W = W_base + s·ΔW with s < 1
moves the model part of the way back toward the base. The base is better at topic search (K5 hit@10 he 50.0% vs our
45.4%, test), ours far better on the link tasks; a scale between may keep most of the link gain and recover topics.

check        merge with s = 1 in memory and confirm it equals the unscaled flagship (models/flagship-14b-ext/final)
merge PCT    write models/flagship-14b-ext-ls<PCT>/final with s = PCT/100 (bfloat16, like the flagship)
adapter PCT  write models/flagship-14b-ext-ls<PCT>/final/adapter: the same adapter weights with lora_alpha × s, so
             base + adapter reproduces the scaled model; verified tensor by tensor (release of the adapter repo)
choose       apply RULE to the KesherBench dev results; prints the chosen model; artifacts/eval/lora_scale/choice.json
             (applied again on the corrected benchmark by scripts/masoretic.py scale-rule → choice_masoretic.json;
             released() gives the released run)
report       reports/F_lora_scale.md

RULE (fixed 2026-11-09, before any scaled model was evaluated):
  candidates s ∈ {0.5, 0.7, 0.85} and the flagship itself (s = 1). Eligible: KesherBench dev score ≥ the flagship's
  minus 0.01. Among the eligible, take the highest topic hit@10 (mean of he and en, dev); ties go to the larger s.
  It replaces the flagship only if its topic hit@10 is higher than the flagship's; otherwise nothing changes.
  Only the chosen model (if not the flagship) runs on the test split. Sefaria bench and Belebele are reported for
  all scales but never used to choose.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from ein_mishpat_ai.eval.kesher import SCORE_SETTINGS, kesher_score, topic_hit10
from ein_mishpat_ai.utils.code_hash import code_hash
from ein_mishpat_ai.utils.io import load_yaml, now_iso, resolve

FLAG = "flagship-14b-ext"
ADAPTER = "models/flagship-14b-ext/final/adapter"
SCALES = {"50": 0.5, "70": 0.7, "85": 0.85}
CHOICE = resolve("artifacts/eval/lora_scale/choice.json")
# the same rule applied again on dev after the Torah text of the benchmark became Masoretic (scripts/masoretic.py
# scale-rule, 2026-11-13); its choice is the release
CHOICE_MT = resolve("artifacts/eval/lora_scale/choice_masoretic.json")
TOL = 0.01


def released() -> str:
    """The run released as ein-mishpat-ai-14b: the rule's choice on the corrected benchmark, else the first choice."""
    return json.loads((CHOICE_MT if CHOICE_MT.exists() else CHOICE).read_text())["model"]


def name(pct: str) -> str:
    return f"{FLAG}-ls{pct}"


def _load(scale: float):
    import torch
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer

    hf_id = load_yaml("configs/eval/models.yaml")["models"]["f2llm-14b"]["hf_id"]
    model = SentenceTransformer(hf_id, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
    peft_model = PeftModel.from_pretrained(model[0].auto_model, str(resolve(ADAPTER)))
    layers = [m for m in peft_model.modules() if hasattr(m, "scaling") and isinstance(m.scaling, dict)]
    if not layers:
        raise SystemExit("no LoRA layers found")
    probe = layers[0]
    before = probe.get_delta_weight("default").float().clone()
    for m in layers:
        for k in m.scaling:
            m.scaling[k] *= scale
    after = probe.get_delta_weight("default").float()
    if not torch.allclose(after, before * scale, rtol=1e-3, atol=1e-6):
        raise SystemExit("scaling did not scale the adapter's weight change")
    peft_model.merge_and_unload()  # in place, as in training and flagship_ext.merge
    if any("lora_" in n for n in model.state_dict()):
        raise SystemExit("LoRA layers still present after merging")
    return model, len(layers)


READING = [
    "- **A smooth trade-off on dev.** As s falls from 1 to 0.5, topic search rises step by step (mean hit@10 41.8% → "
    "43.5% → 44.0% → 45.4%; base 46.0%) and the KesherBench score falls (0.543 → 0.542 → 0.534 → 0.514). On the dev "
    "data of the time s = 0.7 was the largest gain inside the 0.01 limit and was released on 2026-11-10.",
    "- **The rule applied again (2026-11-13).** With the Masoretic Torah text in the benchmark, s = 0.7 costs 0.014 "
    "of the dev score (0.565 → 0.551), outside the 0.01 limit, so it no longer passes its own rule. s = 0.85 was "
    "rebuilt from the base and the adapter (exactly the 2026-11-10 merge) and encoded again: it costs 0.002 and gains "
    "1.5 points of topic search, so the rule now picks it. s = 0.5 was not run again: it lost more than 0.7 on every "
    "earlier measurement. The test split was run for 0.85 only after the choice.",
    "- **Test, 0.85 against the flagship (s = 1):** −0.005 on the KesherBench score (clear but small), topic search "
    "+1.8 points Hebrew and +2.3 English (both clear). The cost sits in commentary → Torah verse and Rashi/Tosafot → "
    "Talmud (−0.014 to −0.019); K1 English → Hebrew rises (+0.010).",
    "- **Test, 0.85 against the earlier release (0.7):** +0.013 on the score (clear; commentary → Torah verse +0.023 "
    "Hebrew, +0.018 English → Hebrew; Rashi/Tosafot → Talmud +0.032), topic search −2.1 / −2.0 points (clear). The "
    "14B is clearly ahead of the 1.7B again on the score (0.564 vs 0.545).",
    "- **Against the base 14B:** score +0.182; topic search English 41.1% vs 39.7% (no clear difference), Hebrew "
    "46.9% vs 49.9% (−3.0 points, at the edge of significance: the interval ends at −0.002 here and at 0.000 in M5's "
    "resampling). The scaling halves the Hebrew topic gap of the "
    "unscaled model (−4.8 points) but does not close it.",
    "- **Side effects (information only):** Sefaria bench 96.8% at every scale from 0.7 to 1; Belebele he→en moves "
    "toward the base (0.891 at s = 1, 0.901 at 0.85, 0.910 at 0.7, base 0.921).",
    "- **Released: s = 0.85** (the rule's choice on the corrected benchmark, 2026-11-13; from "
    "2026-11-10 to 2026-11-13 the staged release was s = 0.7). `ein-mishpat-ai-14b` is the s = 0.85 model; its "
    "adapter repository stores the scale as lora_alpha 27.2 (base + adapter = the merged model, all 442 tensors "
    "identical).",
]


def check() -> None:
    """s = 1 must reproduce the unscaled flagship exactly (same base, same adapter, same merge)."""
    from safetensors import safe_open

    model, n = _load(1.0)
    sd = model[0].auto_model.state_dict()
    final = resolve(f"models/{FLAG}/final")
    index = json.loads((final / "model.safetensors.index.json").read_text())["weight_map"]
    keys = [k for k in index if any(x in k for x in ("layers.0.self_attn.q_proj", "layers.20.mlp.down_proj",
                                                       "layers.39.self_attn.o_proj"))]
    for k in keys:
        with safe_open(str(final / index[k]), framework="pt") as f:
            saved = f.get_tensor(k)
        mine = sd[k.removeprefix("model.")] if k not in sd else sd[k]
        if not (mine.cpu() == saved).all():
            raise SystemExit(f"s = 1 does not reproduce the flagship at {k}")
    print(f"check passed: {n} LoRA layers; s = 1 reproduces the flagship exactly at {len(keys)} tensors")


def merge(pct: str) -> None:
    final = resolve(f"models/{name(pct)}/final")
    if (final / "trained_from.json").exists():
        return
    model, n = _load(SCALES[pct])
    model.save(str(final))
    (final / "trained_from.json").write_text(json.dumps(
        {"adapter": ADAPTER, "scale": SCALES[pct], "lora_layers": n, "finished": now_iso(), "code_hash": code_hash()},
        indent=2))
    print(f"{name(pct)}: saved")


def adapter(pct: str) -> None:
    """The LoRA adapter repo for a scaled model: peft scales by lora_alpha / r, so lora_alpha × s gives s."""
    import shutil

    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from sentence_transformers import SentenceTransformer

    src, final = resolve(ADAPTER), resolve(f"models/{name(pct)}/final")
    dst = final / "adapter"
    if (dst / "verified.json").exists():
        return
    cfg = json.loads((src / "adapter_config.json").read_text())
    if cfg.get("use_rslora") or cfg.get("alpha_pattern") or cfg.get("rank_pattern") or cfg.get("use_dora"):
        raise SystemExit("adapter uses a scaling scheme other than lora_alpha / r")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "adapter_model.safetensors", dst / "adapter_model.safetensors")
    cfg["lora_alpha"] = cfg["lora_alpha"] * SCALES[pct]
    cfg["task_type"] = "FEATURE_EXTRACTION"  # Hugging Face warns on null; merge unchanged (check_lora_task_type.py)
    if cfg["lora_alpha"] / cfg["r"] != SCALES[pct]:
        raise SystemExit("lora_alpha / r does not equal the scale exactly")
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    hf_id = load_yaml("configs/eval/models.yaml")["models"]["f2llm-14b"]["hf_id"]
    model = SentenceTransformer(hf_id, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
    PeftModel.from_pretrained(model[0].auto_model, str(dst)).merge_and_unload()
    sd = model[0].auto_model.state_dict()
    index = json.loads((final / "model.safetensors.index.json").read_text())["weight_map"]
    bad = 0
    for f in sorted(set(index.values())):
        with safe_open(str(final / f), framework="pt") as h:
            for k in h.keys():
                mine = sd[k] if k in sd else sd[k.removeprefix("model.")]
                bad += int(not torch.equal(mine.cpu(), h.get_tensor(k)))
    if bad or len(index) != len(sd):
        raise SystemExit(f"base + scaled adapter differs from {name(pct)} in {bad} tensors ({len(index)} vs {len(sd)})")
    (dst / "verified.json").write_text(json.dumps(
        {"lora_alpha": cfg["lora_alpha"], "r": cfg["r"], "scale": SCALES[pct], "tensors_identical": len(index),
         "checked": now_iso()}, indent=2))
    print(f"{name(pct)}: base + adapter (lora_alpha {cfg['lora_alpha']}) = merged model, all {len(index)} tensors identical")


def dev(model: str) -> dict:
    """Dev results as they were when the scale was chosen (2026-11-10), i.e. with the earlier Torah rendering: the
    scaled models that were not kept (0.5, 0.85) cannot be run again, and the choice is a record, not a new decision."""
    kept = resolve("artifacts/eval/_superseded/kesher_miqra_mevoar_20261112")
    root = kept if kept.exists() else resolve("artifacts/eval/kesher")
    return json.loads((root / "dev" / model / "results.json").read_text())["metrics"]


def choose() -> str:
    if CHOICE.exists():
        return json.loads(CHOICE.read_text())["model"]
    base = dev(FLAG)
    rows = [(1.0, FLAG, base)] + [(s, name(p), dev(name(p))) for p, s in SCALES.items()]
    eligible = [r for r in rows if kesher_score(r[2]) >= kesher_score(base) - TOL]
    best = max(eligible, key=lambda r: (topic_hit10(r[2]), r[0]))
    pick = best[1] if topic_hit10(best[2]) > topic_hit10(base) else FLAG
    rec = {
        "model": pick,
        "rule": "eligible: dev score >= flagship - 0.01; highest dev topic hit@10 (mean he/en); replaces the flagship "
                "only if higher than its topic hit@10",
        "dev": {m: {"scale": s, "score": kesher_score(r), "topic_hit10": topic_hit10(r)} for s, m, r in rows},
        "chosen_at": now_iso(),
        "code_hash": code_hash(),
    }
    CHOICE.parent.mkdir(parents=True, exist_ok=True)
    CHOICE.write_text(json.dumps(rec, indent=2))
    return pick


def _test_pairs(a: str, b: str) -> dict:
    """Paired bootstrap b − a on the test split from the saved top-100 lists (score settings + K5 hit@10)."""
    import polars as pl

    ev = resolve("artifacts/eval/kesher/test")
    bench = resolve("data/bench/kesher")
    rng = np.random.default_rng(13)
    out, sc = {}, {a: np.zeros(1000), b: np.zeros(1000)}
    for ts in [*SCORE_SETTINGS, "K5_topic_sources/he2he", "K5_topic_sources/en2he"]:
        task, setting = ts.split("/")
        meta = json.loads((bench / task / "task.json").read_text())
        corpus = set(pl.read_parquet(bench / "corpora" / f"{meta['corpus']}_{setting[3:]}.parquet",
                                     columns=["id"])["id"].to_list())
        pos: dict[str, set[str]] = {}
        for q, c in pl.read_parquet(bench / task / "qrels_test.parquet").select("query_id", "corpus_id").iter_rows():
            if c in corpus:
                pos.setdefault(q, set()).add(c)
        per = {}
        for m in (a, b):
            z = np.load(ev / m / "topk" / f"{task}__{setting}.npz")
            qs = z["query_ids"].tolist()
            P = [pos[q] for q in qs]
            if ts.startswith("K5"):
                per[m] = np.array([float(any(c in p for c in r[:10])) for r, p in zip(z["corpus_ids"].tolist(), P, strict=True)])
            else:
                per[m] = np.array([sum(1 / np.log2(i + 2) for i, c in enumerate(r[:10]) if c in p)
                                   / sum(1 / np.log2(i + 2) for i in range(min(len(p), 10)))
                                   for r, p in zip(z["corpus_ids"].tolist(), P, strict=True)])
        idx = rng.integers(0, len(per[a]), (1000, len(per[a])))
        d = per[b] - per[a]
        reps = d[idx].mean(1)
        out[ts] = {a: float(per[a].mean()), b: float(per[b].mean()), "diff": float(d.mean()),
                   "ci": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]}
        if not ts.startswith("K5"):
            for m in (a, b):
                sc[m] += per[m][idx].mean(1) / len(SCORE_SETTINGS)
    reps = sc[b] - sc[a]
    out["score"] = {"diff": float(np.mean([out[t]["diff"] for t in SCORE_SETTINGS])),
                    "ci": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]}
    return out


def report() -> None:
    rec = json.loads(CHOICE.read_text())
    first = rec["model"]
    pick = released()
    t = "K5_topic_sources"

    def side(m: str) -> tuple[str, str]:
        sb = resolve(f"artifacts/eval/sefaria_bench/{m}__space/results.json")
        sbr = f"{json.loads(sb.read_text())['metrics']['recall_at_1']:.1%}" if sb.exists() else "—"
        bl = resolve(f"artifacts/eval/regression_clean/{m}/results.json")
        if bl.exists():
            b = json.loads(bl.read_text())["metrics"]
            blr = " / ".join(f"{b[f'{k}/ndcg_at_10']:.3f}" for k in ("belebele_heb", "belebele_eng2heb", "belebele_heb2eng"))
        else:
            blr = "—"
        return sbr, blr

    L = [
        "# Flagship with a scaled LoRA adapter (topic search)",
        "",
        f"Generated {now_iso()} by `scripts/lora_scale.py` (code hash `{code_hash()}`; `src/` unchanged since the M5 "
        "harness). Follow-up check (2026-11-09).",
        "",
        "**What:** the flagship's LoRA adapter is a learned change added to F2LLM-v2-14B. Here it is multiplied by a "
        "scale s before merging, so the model moves part of the way back toward the base, which is better at topic "
        "search. Checked first: s = 1 reproduces the unscaled flagship exactly.",
        "",
        "**Rule (fixed before any scaled model was evaluated):** eligible scales keep the KesherBench dev score within "
        "0.01 of the flagship; among them the highest topic hit@10 (mean of Hebrew and English, dev) wins, and it "
        "replaces the flagship only if that is higher than the flagship's. Only the chosen model runs on test. "
        "Sefaria bench and Belebele are information only.",
        "",
        f"**Released: `{pick}`.**" + (f" The first choice (2026-11-10) was `{first}`; section 2 says why it changed."
                                       if pick != first else ""),
        "",
        "## 1. Dev, first choice (2026-11-10)",
        "",
        "As it was when the scale was first chosen: the Torah passages still in the earlier modern-Hebrew rendering.",
        "",
        "| scale s | model | KesherBench score | topic hit@10 he | topic hit@10 en | mean | Sefaria R@1 | Belebele he / en→he / he→en |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    rows = [(0.0, "f2llm-14b"), *sorted(((v["scale"], m) for m, v in rec["dev"].items()), key=lambda x: x[0])]
    for s, m in rows:
        r = dev(m)
        sbr, blr = side(m)
        lab = "0 (base)" if m == "f2llm-14b" else f"{s:g}"
        mark = " first choice" if m == first else ""
        L.append(f"| {lab} | `{m}`{mark} | {kesher_score(r):.3f} | {r[f'{t}/he2he/hit_at_10']:.1%} | "
                 f"{r[f'{t}/en2he/hit_at_10']:.1%} | {topic_hit10(r):.1%} | {sbr} | {blr} |")
    L += [""]
    if CHOICE_MT.exists():
        mt = json.loads(CHOICE_MT.read_text())
        L += ["## 2. Dev, the same rule on the corrected benchmark (2026-11-13)", "",
              "The Torah and the Five Megillot in the Masoretic text (`scripts/masoretic.py scale-rule`, "
              "`artifacts/eval/lora_scale/choice_masoretic.json`). Candidates s = 1, 0.85, 0.7; s = 0.85 was rebuilt "
              "from the base and the adapter and encoded again; s = 0.5 was not run again (it lost more than 0.7 on "
              "every earlier measurement).", "",
              "| scale s | model | KesherBench score | topic hit@10 he | topic hit@10 en | mean | eligible (score ≥ s = 1 − 0.01) |",
              "|---:|---|---:|---:|---:|---:|---|"]
        cur = resolve("artifacts/eval/kesher/dev")
        for m, v in sorted(mt["dev"].items(), key=lambda x: -x[1]["scale"]):
            r = json.loads((cur / m / "results.json").read_text())["metrics"]
            mark = " **chosen**" if m == mt["model"] else ""
            L.append(f"| {v['scale']:g} | `{m}`{mark} | {v['score']:.3f} | {r[f'{t}/he2he/hit_at_10']:.1%} | "
                     f"{r[f'{t}/en2he/hit_at_10']:.1%} | {v['topic_hit10']:.1%} | {'yes' if v['eligible'] else 'no'} |")
        L += ["", f"**Chosen: `{mt['model']}`.** Released as ein-mishpat-ai-14b (2026-11-13).", ""]
    if pick != FLAG:
        sbr, blr = side(pick)
        L += [f"## 3. Test split, released model (`{pick}`; paired bootstrap, 1,000 resamples over queries)", "",
              f"Sefaria bench R@1 {sbr}; Belebele he / en→he / he→en {blr}.", ""]
        others = [(FLAG, "flagship (s = 1)"), *([(first, f"first choice ({first})")] if first != pick else []),
                  ("f2llm-14b", "base F2LLM-v2-14B")]
        for other, lab_o in others:
            pr = _test_pairs(other, pick)
            L += [f"Released minus {lab_o}:", "", f"| setting | {lab_o} | released | difference |", "|---|---:|---:|---:|"]
            for ts, v in pr.items():
                if ts == "score":
                    continue
                st = " ✱" if v["ci"][0] > 0 or v["ci"][1] < 0 else ""
                fmt = (lambda x: f"{x:.1%}") if ts.startswith("K5") else (lambda x: f"{x:.3f}")
                lab = ts.replace("/", " ") + (" (hit@10)" if ts.startswith("K5") else " (nDCG@10)")
                L.append(f"| {lab} | {fmt(v[other])} | {fmt(v[pick])} | {v['diff']:+.3f} "
                         f"[{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]{st} |")
            s = pr["score"]
            st = " ✱" if s["ci"][0] > 0 or s["ci"][1] < 0 else ""
            L += [f"| **KesherBench score** | | | {s['diff']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]{st} |", ""]
        L += ["✱ = the 95% interval excludes 0. It covers resampled queries, not the run-to-run variation of "
              "training (one seed).", ""]
    else:
        L += ["No scale passed the rule, so the flagship stays as it is and nothing ran on test.", ""]
    L += ["## Reading (written after the results; updated 2026-11-13)", "", *READING, ""]
    resolve("reports/F_lora_scale.md").write_text("\n".join(L))
    print("reports/F_lora_scale.md")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "check":
        check()
    elif cmd == "merge":
        merge(sys.argv[2])
    elif cmd == "adapter":
        adapter(sys.argv[2])
    elif cmd == "choose":
        print(choose())
    elif cmd == "report":
        report()
    else:
        raise SystemExit(__doc__)
