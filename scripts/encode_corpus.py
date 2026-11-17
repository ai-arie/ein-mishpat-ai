"""usage: python scripts/encode_corpus.py MODEL TABLE.parquet OUT_DIR [--shard k --num-shards n]
[--prefix "Instruct: …\\nQuery: "] [--max-length 512] [--batch-size 64]"""

import argparse
from pathlib import Path

from ein_mishpat_ai.eval.encode import encode_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("table")
    ap.add_argument("out")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()
    encode_table(
        a.model,
        Path(a.table),
        Path(a.out),
        a.shard,
        a.num_shards,
        a.prefix.encode().decode("unicode_escape") if a.prefix else "",
        a.max_length,
        a.batch_size,
    )


if __name__ == "__main__":
    main()
