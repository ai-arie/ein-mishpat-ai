"""M1 data pipeline. usage: python scripts/build_data.py {versions,segments,links,report,all}"""

import argparse

from ein_mishpat_ai.data.download_sefaria import load_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["versions", "segments", "links", "report", "all"])
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    cfg = load_cfg()
    stages = ["versions", "segments", "links", "report"] if args.stage == "all" else [args.stage]
    for st in stages:
        if st == "versions":
            from ein_mishpat_ai.data.segments import build_versions

            build_versions(cfg)
        elif st == "segments":
            from ein_mishpat_ai.data.segments import build_segments

            build_segments(cfg, workers=args.workers)
        elif st == "links":
            from ein_mishpat_ai.data.links import build_links

            build_links(cfg, workers=args.workers)
        elif st == "report":
            from ein_mishpat_ai.data.profile_m1 import write_report

            write_report(cfg)


if __name__ == "__main__":
    main()
