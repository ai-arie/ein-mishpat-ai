"""CLI: python scripts/download_sefaria.py {meta,texts,status} [--pilot|--all]"""

from __future__ import annotations

import argparse
import json

from ein_mishpat_ai.data import download_sefaria as ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("meta")
    t = sub.add_parser("texts")
    g = t.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--all", action="store_true")
    t.add_argument("--verify-sha", action="store_true", help="re-hash kept files before resuming")
    sub.add_parser("status")
    sub.add_parser("bucket-sizes")
    args = ap.parse_args()
    cfg = ds.load_cfg(args.config)
    if args.cmd == "meta":
        ds.run_meta(cfg)
    elif args.cmd == "texts":
        ds.run_texts(cfg, pilot=args.pilot, verify_sha=args.verify_sha)
    elif args.cmd == "bucket-sizes":
        print(len(ds.bucket_sizes(cfg)))
    else:
        print(json.dumps(ds.status(cfg), indent=2))


if __name__ == "__main__":
    main()
