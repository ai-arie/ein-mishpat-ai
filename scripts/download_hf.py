"""CLI: python scripts/download_hf.py"""

from __future__ import annotations

import argparse

from ein_mishpat_ai.data.download_hf import run_hf
from ein_mishpat_ai.data.download_sefaria import load_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    run_hf(load_cfg(args.config))


if __name__ == "__main__":
    main()
