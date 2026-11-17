"""M3a teacher pass. usage: python scripts/teacher.py {prepare,mine} [--task TASK]

prepare: input tables + data/train/teacher/{jobs,mine_jobs}.yaml (then run_parallel.py on them)
mine:    hard negatives + teacher scores for one task (needs the encodings)
"""

import argparse
import json

from ein_mishpat_ai.data.download_sefaria import load_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "mine"])
    ap.add_argument("--task")
    a = ap.parse_args()
    cfg = load_cfg()
    if a.stage == "prepare":
        from ein_mishpat_ai.train.teacher import prepare

        print(json.dumps(prepare(cfg), indent=2))
    else:
        from ein_mishpat_ai.train.teacher import mine

        print(json.dumps(mine(cfg, a.task), indent=2))


if __name__ == "__main__":
    main()
