"""usage: python scripts/eval_kesher.py MODEL|bm25 [--split dev|test]"""

import argparse
import json

from ein_mishpat_ai.eval.kesher import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--split", default=None)
    ap.add_argument("--config", default="configs/eval/kesher.yaml")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    rec = evaluate(a.model, a.config, a.split, a.only)
    print(json.dumps(rec["n_queries"], indent=2))


if __name__ == "__main__":
    main()
