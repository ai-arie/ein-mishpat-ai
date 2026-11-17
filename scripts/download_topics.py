"""CLI: python scripts/download_topics.py"""

from __future__ import annotations

import argparse
import json

from ein_mishpat_ai.data.download_sefaria import load_cfg
from ein_mishpat_ai.data.download_topics import run_topics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    print(json.dumps(run_topics(load_cfg(args.config))))


if __name__ == "__main__":
    main()
