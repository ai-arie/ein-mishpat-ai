"""KesherBench test, counting only the answer passages that lie in held-out (test) works.

usage: python scripts/strict_gold.py   → artifacts/eval/strict_gold/test.json, reports/F_heldout_answers.md

Why (found 2026-11-11): a query is a test query when at least one of its gold sources is in a test work, and it keeps
all its gold sources (bench/kesher.py). So part of the gold passages lie in works that were used in training as text
(never as a pair with the query, and never the query passage itself; tests/test_leakage.py). This script measures how
much that matters. CPU only: it reads the top-100 lists the harness saved for the test split and encodes nothing.

Strict metric: positives = gold passages in test works. The other gold passages (training or dev works) are taken
out of the ranked list before nDCG@10 is computed, so they are neither rewarded nor counted as errors. Every test
query has at least one gold passage in a test work, so no query is lost. The all-gold numbers are recomputed too and
must equal the harness's results.json. 95% bootstrap over queries (1,000 resamples, seed 13); differences are paired.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from ein_mishpat_ai.eval.kesher import SCORE_SETTINGS
from ein_mishpat_ai.utils.code_hash import code_hash
from ein_mishpat_ai.utils.io import load_yaml, now_iso, resolve

MODELS = [
    ("flagship-14b-ext-ls85", "ein-mishpat-ai-14b"),
    ("student-1.7b-full-d3b", "ein-mishpat-ai-1.7b"),
    ("student-0.6b-full-d3b", "ein-mishpat-ai-0.6b"),
    ("f2llm-14b", "F2LLM-v2-14B"),
    ("f2llm-8b", "F2LLM-v2-8B"),
    ("qwen3-8b", "Qwen3-Embedding-8B"),
    ("f2llm-1.7b", "F2LLM-v2-1.7B"),
    ("f2llm-0.6b", "F2LLM-v2-0.6B"),
]
OURS, BEST = [m for m, _ in MODELS[:3]], "f2llm-14b"
EXTRA = ["K3_gemara_mishnah/he2he", "K3_gemara_mishnah/en2he"]  # outside the score, shown for completeness
EV, BENCH = resolve("artifacts/eval/kesher/test"), resolve("data/bench/kesher")
OUT = resolve("artifacts/eval/strict_gold/test.json")
B, SEED = 1000, 13

READING = [
    "- **The lead does not come from answers in training works.** Counting only answers in held-out works, the "
    "KesherBench score of every model falls a little (the task gets harder: fewer correct answers per query), and our "
    "models stay ahead of the best open model tested by almost the same margin.",
    "- **Where it matters:** Mishneh Torah → Talmud (K1), where about half of the answer passages are in training "
    "tractates. The commentary tasks (K2a, K2b) are unchanged: their answers are all in held-out works.",
    "- **What was never in training in any case:** the query passages, every query–answer pair, the held-out works "
    "and the Sefaria benchmark's passages (unit tests).",
]


def ndcg(row: list[str], p: set[str]) -> float:
    d = sum(1 / np.log2(r + 2) for r, c in enumerate(row[:10]) if c in p)
    return d / sum(1 / np.log2(r + 2) for r in range(min(len(p), 10)))


def main() -> None:
    t = pl.read_parquet(resolve(load_yaml("configs/data.yaml")["processed"]["texts"]), columns=["ref", "split"])
    split_of = dict(zip(t["ref"].to_list(), t["split"].to_list(), strict=True))
    rng = np.random.default_rng(SEED)
    res: dict = {"settings": {}, "models": {m: {} for m, _ in MODELS}, "gold": {}}
    reps = {m: {"all": np.zeros(B), "strict": np.zeros(B)} for m, _ in MODELS}

    for task in sorted(p.name for p in BENCH.iterdir() if (p / "qrels_test.parquet").exists()):
        g = set(pl.read_parquet(BENCH / task / "qrels_test.parquet")["corpus_id"].to_list())
        by = {s: sum(split_of.get(x) == s for x in g) for s in ("test", "dev", "train")}
        res["gold"][task] = {"passages": len(g), **by}

    for ts in [*SCORE_SETTINGS, *EXTRA]:
        task, setting = ts.split("/")
        meta = json.loads((BENCH / task / "task.json").read_text())
        corpus = set(pl.read_parquet(BENCH / "corpora" / f"{meta['corpus']}_{setting[3:]}.parquet", columns=["id"])["id"].to_list())
        pos: dict[str, set[str]] = {}
        for q, c in pl.read_parquet(BENCH / task / "qrels_test.parquet").select("query_id", "corpus_id").iter_rows():
            if c in corpus:
                pos.setdefault(q, set()).add(c)
        idx = None
        for m, _ in MODELS:
            z = np.load(EV / m / "topk" / f"{task}__{setting}.npz", allow_pickle=False)
            qs, rows = z["query_ids"].tolist(), z["corpus_ids"].tolist()
            a, s = [], []
            for q, row in zip(qs, rows, strict=True):
                p = pos[q]
                pt = {c for c in p if split_of.get(c) == "test"}
                if not pt:
                    raise SystemExit(f"{ts} {q}: no gold passage in a test work")
                a.append(ndcg(row, p))
                s.append(ndcg([c for c in row if c not in p - pt], pt))
            a, s = np.array(a), np.array(s)
            harness = json.loads((EV / m / "results.json").read_text())["metrics"][f"{ts}/ndcg_at_10"]
            if abs(a.mean() - harness) > 1e-9:
                raise SystemExit(f"{m} {ts}: recomputed nDCG {a.mean()} differs from the harness's {harness}")
            if idx is None:
                idx = rng.integers(0, len(qs), (B, len(qs)))
                npos = [len(pos[q]) for q in qs]
                ntest = [sum(split_of.get(c) == "test" for c in pos[q]) for q in qs]
                res["settings"][ts] = {"n": len(qs), "gold_per_query": float(np.mean(npos)),
                                       "gold_per_query_test": float(np.mean(ntest)),
                                       "share_test": float(sum(ntest) / sum(npos))}
            res["models"][m][ts] = {"all": float(a.mean()), "strict": float(s.mean())}
            if ts in SCORE_SETTINGS:
                reps[m]["all"] += a[idx].mean(1) / len(SCORE_SETTINGS)
                reps[m]["strict"] += s[idx].mean(1) / len(SCORE_SETTINGS)

    for m, _ in MODELS:
        r = res["models"][m]
        sc = {k: float(np.mean([r[ts][k] for ts in SCORE_SETTINGS])) for k in ("all", "strict")}
        r["score"] = {**sc, "strict_ci": [float(x) for x in np.percentile(reps[m]["strict"], [2.5, 97.5])]}
        for k in ("all", "strict"):
            d = reps[m][k] - reps[BEST][k]
            r["score"][f"minus_best_{k}"] = float(sc[k] - np.mean([res["models"][BEST][ts][k] for ts in SCORE_SETTINGS]))
            r["score"][f"minus_best_{k}_ci"] = [float(x) for x in np.percentile(d, [2.5, 97.5])]
    res.update(best_existing=BEST, code_hash=code_hash(), created=now_iso())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    report(res)


def report(res: dict) -> None:
    M, names = res["models"], dict(MODELS)
    f3 = lambda x: f"{x:.3f}"  # noqa: E731
    L = [
        "# KesherBench test: counting only answers in held-out works",
        "",
        f"Generated {now_iso()} by `scripts/strict_gold.py` (code hash `{res['code_hash']}`) from the rankings the "
        "harness saved for the test split; no model was run again.",
        "",
        "**Why.** A query is a test query when at least one of its gold sources is in a test work, and it keeps all its "
        "gold sources. So some answer passages lie in works that were used in training as text. No query passage and no "
        "query–answer pair was ever trained on, but the models have seen those passages in other pairs. This report "
        "checks whether that explains any of the results.",
        "",
        "**Strict metric.** Positives = gold passages in test works only. Gold passages in training or dev works are "
        "removed from each ranked list before nDCG@10 is computed, so they neither help nor count as errors. Every test "
        "query has at least one gold passage in a test work, so the queries are the same. [low, high] = 95% bootstrap "
        "interval over queries; ✱ = the interval of a paired difference excludes 0.",
        "",
        "## 1. Where the answer passages are (test split)",
        "",
        "| task | answer passages | in test works | in dev works (never trained) | in training works |",
        "|---|---:|---:|---:|---:|",
    ]
    for task, g in res["gold"].items():
        L.append(f"| {task} | {g['passages']:,} | {g['test']:,} | {g['dev']:,} | {g['train']:,} ({g['train'] / g['passages']:.0%}) |")
    L += [
        "",
        "K5 (topic → sources) is built differently: its topics are held out, and their sources come from the whole "
        "library. It is outside the KesherBench score.",
        "",
        "## 2. KesherBench score (mean nDCG@10 over the 8 settings)",
        "",
        f"| model | all answers (reported) | answers in test works only | lead over {names[res['best_existing']]}: all → strict |",
        "|---|---:|---:|---:|",
    ]
    for m, nm in MODELS:
        s = M[m]["score"]
        lead = "—"
        if m != res["best_existing"]:
            lo, hi = s["minus_best_strict_ci"]
            lead = f"{s['minus_best_all']:+.3f} → {s['minus_best_strict']:+.3f} [{lo:+.3f}, {hi:+.3f}]{' ✱' if lo > 0 or hi < 0 else ''}"
        L.append(f"| {nm} | {f3(s['all'])} | {f3(s['strict'])} [{f3(s['strict_ci'][0])}, {f3(s['strict_ci'][1])}] | {lead} |")
    L += ["", "## 3. Per setting (nDCG@10: all answers → answers in test works only)", "",
          "| setting | n | answers in test works | " + " | ".join(nm for _, nm in MODELS[:4]) + " |",
          "|---|---:|---:|" + "---:|" * 4]
    for ts, v in res["settings"].items():
        cells = [f"{f3(M[m][ts]['all'])} → {f3(M[m][ts]['strict'])}" for m, _ in MODELS[:4]]
        L.append(f"| {ts.replace('/', ' ')}{'' if ts in SCORE_SETTINGS else ' (not in the score)'} | {v['n']} | "
                 f"{v['share_test']:.0%} | " + " | ".join(cells) + " |")
    L += ["", "## Reading (written after the results, 2026-11-11)", "", *READING, ""]
    resolve("reports/F_heldout_answers.md").write_text("\n".join(L))
    print("reports/F_heldout_answers.md")


if __name__ == "__main__":
    main()
