"""M3b synthetic translations. usage: python scripts/translate.py {trial,score}

trial: translate the 50-segment trial sample with every setting in configs/gen.yaml (API, no GPU)
score: teacher similarity to the human translations (one GPU)
batch / batch-score: several consecutive segments per request vs one per request
d3a-select / d3a-run: full synthetic translation run (src/ein_mishpat_ai/gen/d3a.py; exit 3 = paused)
d3a-redo: re-translate early-block batches with Hebrew-letter numerals (gematria digits)
d3b-select / d3b-run: synthetic search queries (src/ein_mishpat_ai/gen/d3b.py; exit 3 = paused)
d3b-ext-select / d3b-ext-run: the remaining 200k passages (exit 4 = stopped at the token budget)
"""

import argparse
import json

from ein_mishpat_ai.data.download_sefaria import load_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "stage",
        choices=[
            "trial",
            "score",
            "batch",
            "batch-score",
            "d3a-select",
            "d3a-run",
            "d3a-redo",
            "d3b-select",
            "d3b-run",
            "d3b-ext-select",
            "d3b-ext-run",
        ],
    )
    a = ap.parse_args()
    if a.stage == "trial":
        from ein_mishpat_ai.gen.translate import run_trial

        run_trial(load_cfg())
    elif a.stage == "score":
        from ein_mishpat_ai.gen.translate import score_trial

        print(json.dumps(score_trial(), indent=2))
    elif a.stage == "batch":
        from ein_mishpat_ai.gen.translate import run_batch_test

        run_batch_test(load_cfg())
    elif a.stage == "d3a-select":
        from ein_mishpat_ai.gen.d3a import select

        print(json.dumps(select(load_cfg()), indent=2, ensure_ascii=False))
    elif a.stage == "d3a-redo":
        from ein_mishpat_ai.gen.d3a import redo

        print(json.dumps(redo(load_cfg()), indent=2))
    elif a.stage == "d3a-run":
        from ein_mishpat_ai.gen.d3a import run

        raise SystemExit(run(load_cfg()))
    elif a.stage == "d3b-select":
        from ein_mishpat_ai.gen.d3b import select as d3b_select

        print(json.dumps(d3b_select(load_cfg()), indent=2, ensure_ascii=False))
    elif a.stage == "d3b-ext-select":
        from ein_mishpat_ai.gen.d3b import select as d3b_select

        print(json.dumps(d3b_select(load_cfg(), "d3b_ext"), indent=2, ensure_ascii=False))
    elif a.stage == "d3b-ext-run":
        from ein_mishpat_ai.gen.d3b import run as d3b_run

        raise SystemExit(d3b_run("d3b_ext"))
    elif a.stage == "d3b-run":
        from ein_mishpat_ai.gen.d3b import run as d3b_run

        raise SystemExit(d3b_run())
    else:
        from ein_mishpat_ai.gen.translate import score_batch_test

        print(json.dumps(score_batch_test(), indent=2))


if __name__ == "__main__":
    main()
