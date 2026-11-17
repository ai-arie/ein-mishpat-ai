"""usage: python scripts/eval_retrieval.py MODEL --config configs/eval/<task>.yaml"""

import argparse
import json

from ein_mishpat_ai.eval.retrieval import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    rec = evaluate(a.model, a.config)
    print(json.dumps({"model": a.model, **rec["metrics"], **rec["n_queries"]}, indent=2))


if __name__ == "__main__":
    main()
