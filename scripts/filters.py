"""M3 teacher filters. usage: python scripts/filters.py {d3a-prepare,d3a-score,d1-score}

d3a-prepare: query tables + data/gen/d3a/roundtrip/jobs.yaml for finished D3a blocks (then run_parallel.py)
d3a-score:   round-trip rank of each synthetic translation's source (one GPU, needs the encodings)
d1-score:    teacher rank/score of the paired side for every D1 bitext pair (one GPU)
d1-keep:     `keep` column from configs/teacher.yaml d1_filter (CPU)
d3a-keep:    `keep` column in the D3a round-trip scores from d3a_filter (CPU, after d3a-score)
d3a-mine:    hard negatives + teacher scores for D3a from the round-trip encodings (one GPU)
d3b-prepare: query tables + encoding jobs for the D3b search queries (then run_parallel.py)
d3b-mine:    D3b round-trip rank + hard negatives (one GPU)
d3b-ext-prepare / d3b-ext-mine: the same for the remaining 200k passages
"""

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "stage",
        choices=[
            "d3a-prepare",
            "d3a-score",
            "d1-score",
            "d1-keep",
            "d3a-keep",
            "d3a-mine",
            "d3b-prepare",
            "d3b-mine",
            "d3b-ext-prepare",
            "d3b-ext-mine",
        ],
    )
    a = ap.parse_args()
    from ein_mishpat_ai.train import filters

    fn = {
        "d3a-prepare": filters.d3a_prepare,
        "d3a-score": filters.d3a_score,
        "d1-score": filters.d1_score,
        "d1-keep": filters.d1_keep,
        "d3a-keep": filters.d3a_keep,
        "d3a-mine": filters.d3a_mine,
        "d3b-prepare": filters.d3b_prepare,
        "d3b-mine": filters.d3b_mine,
        "d3b-ext-prepare": lambda: filters.d3b_prepare("d3b_ext"),
        "d3b-ext-mine": lambda: filters.d3b_mine("d3b_ext"),
    }
    print(json.dumps(fn[a.stage](), indent=2))


if __name__ == "__main__":
    main()
