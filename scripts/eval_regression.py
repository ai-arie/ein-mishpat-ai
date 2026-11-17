"""usage: python scripts/eval_regression.py [--config CFG] prepare | MODEL ...   (skips models already evaluated)

--config defaults to configs/eval/regression.yaml; configs/eval/regression_clean.yaml is the clean check (M6)."""

import json
import sys

from ein_mishpat_ai.eval.regression import CONFIG, evaluate, prepare
from ein_mishpat_ai.utils.io import load_yaml, resolve

args = sys.argv[1:]
cfg_path = CONFIG
if args[:1] == ["--config"]:
    cfg_path, args = args[1], args[2:]
if args[0] == "prepare":
    print(json.dumps(prepare(cfg_path), indent=2))
else:
    out = resolve(load_yaml(cfg_path)["out_dir"])
    for model in args:
        if (out / model / "results.json").exists():
            print(f"{model}: already evaluated")
            continue
        rec = evaluate(model, cfg_path)
        print(model, json.dumps({k: round(v, 4) for k, v in rec["metrics"].items() if "ndcg" in k}))
