"""Write run_parallel.py job files to encode the KesherBench corpora and evaluate MODEL(s) — the
same commands as configs/jobs/m2_kesher_{encode,eval}.yaml (one harness for every model).

usage: python scripts/kesher_jobs.py NAME MODEL [MODEL ...] [--split dev]
→ configs/jobs/NAME_encode.yaml, configs/jobs/NAME_eval.yaml
"""

import argparse
import math

import polars as pl
import yaml

from ein_mishpat_ai.utils.io import load_yaml, resolve

ROWS_PER_SHARD = 80_000  # the 638k K5 pool in 8 shards so every GPU stays busy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out-dir", default="configs/jobs", help="scripts/jobs keeps configs/ (code hash) unchanged")
    a = ap.parse_args()
    kc = load_yaml("configs/eval/kesher.yaml")
    corpora = sorted((resolve(kc["bench_dir"]) / "corpora").glob("*.parquet"))
    enc, ev = [], []
    models = load_yaml("configs/eval/models.yaml")["models"]
    for m in a.models:
        dp = models.get(m, {}).get("doc_prompt")
        pre = f" --prefix '{dp}'" if dp else ""
        for f in corpora:
            n = pl.read_parquet(f, columns=["id"]).height
            shards = max(1, math.ceil(n / ROWS_PER_SHARD))
            for s in range(shards):
                enc.append(
                    {
                        "name": f"enc-{m}-{f.stem}-s{s}",
                        "cmd": f"python scripts/encode_corpus.py {m} {kc['bench_dir']}/corpora/{f.name} "
                        f"{kc['emb_dir']}/{m}/{f.stem} --shard {s} --num-shards {shards} "
                        f"--batch-size {kc['batch_size']}{pre}",
                        "gpus": 1,
                    }
                )
        ev.append(
            {
                "name": f"eval-{m}",
                "cmd": f"python scripts/eval_kesher.py {m} --split {a.split}",
                "gpus": 1,
            }
        )
    for kind, jobs in (("encode", enc), ("eval", ev)):
        path = resolve(f"{a.out_dir}/{a.name}_{kind}.yaml")
        path.write_text(yaml.safe_dump({"batch": f"{a.name}_{kind}", "jobs": jobs}, width=1000))
        print(path, len(jobs))


if __name__ == "__main__":
    main()
