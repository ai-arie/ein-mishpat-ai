"""M4 training. usage: python scripts/train.py mix configs/train/mix_d3b_all.yaml
torchrun --nproc_per_node 8 scripts/train.py fit CONFIG   (configs/train/*.yaml; resumes from the last checkpoint)
"""

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["mix", "fit"])
    ap.add_argument("config", nargs="?")
    a = ap.parse_args()
    if a.stage == "mix":
        from ein_mishpat_ai.train.mix import build

        print(json.dumps(build(a.config or "configs/train/mix_d3b_all.yaml"), indent=2, ensure_ascii=False))
    else:
        from ein_mishpat_ai.train.contrastive import fit

        fit(a.config)


if __name__ == "__main__":
    main()
