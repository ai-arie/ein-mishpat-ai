"""usage: python scripts/eval_sefaria_bench.py <model-name from configs/eval/models.yaml>"""

import argparse
import json

from ein_mishpat_ai.eval.sefaria_bench import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--config", default="configs/eval/sefaria_bench.yaml")
    args = ap.parse_args()
    rec = evaluate(args.model, args.config)
    print(json.dumps({"model": args.model, **rec["metrics"], **rec["model_info"]}, indent=2))


if __name__ == "__main__":
    main()
