"""M2 data. usage: python scripts/build_m2.py {leakage,texts,kesher,pairs,all}"""

import argparse
import json

from ein_mishpat_ai.data.download_sefaria import load_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["leakage", "texts", "pairs", "kesher", "all"])
    a = ap.parse_args()
    cfg = load_cfg()
    stages = ["leakage", "texts", "kesher", "pairs"] if a.stage == "all" else [a.stage]
    for st in stages:
        if st == "leakage":
            from ein_mishpat_ai.data.leakage import build_exclusion

            print(json.dumps(build_exclusion(cfg), indent=2))
        elif st == "texts":
            from ein_mishpat_ai.data.pairs import build_texts

            build_texts(cfg)
        elif st == "pairs":
            from ein_mishpat_ai.data.pairs import build_pairs

            print(json.dumps(build_pairs(cfg), indent=2))
        elif st == "kesher":
            from ein_mishpat_ai.bench.kesher import build_all

            print(json.dumps(build_all(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
