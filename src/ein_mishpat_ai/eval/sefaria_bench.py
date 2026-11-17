"""Sefaria Rabbinic Embedding Bench (project rule 5 (README): reproduce before you trust).

Evaluation only: the pairs are never written into training data (rule 1, rule 3 exception).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ein_mishpat_ai.eval.encoder import Encoder, model_spec
from ein_mishpat_ai.eval.metrics import (
    space_bitext_accuracy,
    space_retrieval_metrics,
    space_true_ranks,
)
from ein_mishpat_ai.eval.record import write_run
from ein_mishpat_ai.utils.io import load_yaml, resolve
from ein_mishpat_ai.utils.seed import set_seed


def load_pairs(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(resolve(path).read_text())


def evaluate(model_name: str, cfg_path: str = "configs/eval/sefaria_bench.yaml") -> dict[str, Any]:
    cfg = load_yaml(cfg_path)
    set_seed(cfg["seed"])
    pairs = load_pairs(cfg["data"])
    if cfg["protocol"] != "space":
        raise ValueError(f"unknown protocol {cfg['protocol']}")
    # Space protocol: no prompt for models outside its curated list (all of ours).
    spec = model_spec(model_name, prompt_style="none", max_length=cfg["max_length"])
    spec.batch_size = cfg["batch_size"]

    t0 = time.time()
    enc = Encoder(spec)
    load_s = time.time() - t0
    he = enc.encode([p["he"] for p in pairs], chunked=True)
    en = enc.encode([p["en"] for p in pairs], chunked=True)
    enc_s = time.time() - t0 - load_s

    sim = he @ en.T
    metrics = space_retrieval_metrics(sim, cfg["k_values"])
    acc, avg_true, avg_rand = space_bitext_accuracy(
        sim, cfg["bitext_negatives"], cfg["bitext_seed"]
    )
    metrics.update(
        bitext_accuracy=acc, avg_true_pair_similarity=avg_true, avg_random_pair_similarity=avg_rand
    )
    # Per category: queries of that category, full 3,708 pool (the Space's own per-category
    # function uses a per-category pool instead; ours is labelled pool=full).
    ranks = space_true_ranks(sim)
    cats = np.array([p.get("category", "Unknown") for p in pairs])
    per_cat = {
        c: {"n": int((cats == c).sum()), "recall_at_1": float(np.mean(ranks[cats == c] < 1))}
        for c in sorted(set(cats))
    }

    out = resolve(cfg["out_dir"]) / f"{model_name}__{cfg['protocol']}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "he.npy", he.astype(np.float16))
    np.save(out / "en.npy", en.astype(np.float16))
    np.save(out / "ranks.npy", ranks.astype(np.int32))
    config = {**cfg, "model": model_name, "hf_id": spec.hf_id, "prompt_style": spec.prompt_style}
    return write_run(
        out,
        experiment=cfg["name"],
        run_name=f"{model_name}__{cfg['protocol']}",
        config=config,
        metrics=metrics,
        extra={
            "num_pairs": len(pairs),
            "categories": dict(Counter(cats.tolist())),
            "per_category_pool_full": per_cat,
            "model_info": {
                "dim": enc.dim,
                "n_params": enc.n_params,
                "max_seq_length": enc.model.max_seq_length,
                "load_seconds": round(load_s, 1),
                "encode_seconds": round(enc_s, 1),
            },
        },
    )
