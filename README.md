# ein-mishpat-ai

Open embedding models that find Jewish sources **by meaning** across Hebrew, Aramaic and English, and **KesherBench**, a
benchmark for finding *related* sources, not just translations.

**A retrieval aid, not a halachic authority.** The models suggest texts that may be relevant; they decide nothing.

| | |
|---|---|
| Models | [ein-mishpat-ai-0.6b](https://huggingface.co/ai-arie/ein-mishpat-ai-0.6b) · [ein-mishpat-ai-1.7b](https://huggingface.co/ai-arie/ein-mishpat-ai-1.7b) · [ein-mishpat-ai-14b](https://huggingface.co/ai-arie/ein-mishpat-ai-14b) ([LoRA adapter](https://huggingface.co/ai-arie/ein-mishpat-ai-14b-lora)) |
| Benchmark | [KesherBench](https://huggingface.co/datasets/ai-arie/KesherBench) |
| Collection | [ein-mishpat-ai](https://huggingface.co/collections/ai-arie/ein-mishpat-ai-6afc0fa20fb9a883001d53bd) (models and benchmark on one page) |
| Results | on each model card (with 95% intervals) and on the KesherBench card |

## What it does

- **Translation matching** (Sefaria's [Rabbinic Embedding Bench](https://huggingface.co/spaces/Sefaria/Rabbinic-Embedding-Bench)):
  the 14B and 1.7B models reach 96.8% and 96.7% Recall@1, ahead of the published #1 (Gemini Embedding 001, 93.9%).
- **Finding related sources** (KesherBench): the Talmud sources of a Mishneh Torah ruling, the verse a commentary
  explains, Bavli ↔ Yerushalmi parallels, the key sources of a topic. Even the 0.6B model beats F2LLM-v2-14B, the best
  of the open models tested, which is 23 times larger.
- **Weak spots:** short topic searches (adding keyword search helps the 0.6B model there, but lowers the overall
  KesherBench score), very short queries, and general non-Jewish
  text (use the base model there). Details in the model cards.

## Quick start

```bash
pip install sentence-transformers
python scripts/example_search.py            # indexes three short passages and runs three searches
```

The same in a few lines:

```python
import re
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("ai-arie/ein-mishpat-ai-1.7b")

def strip_marks(text):   # remove vowel points and cantillation first (see "Before you index")
    text = re.sub("[\u0591-\u05AF\u05B0-\u05BD\u05BF\u05C1\u05C2\u05C4\u05C5\u05C7]", "", text)
    return " ".join(text.replace("\u05BE", " ").replace("\u05C0", " ").split())

passages = ["...", "..."]                                         # your Hebrew/Aramaic or English texts
index = model.encode([strip_marks(p) for p in passages])          # once; passages get no instruction
query = model.encode(["How many witnesses are needed to convict in a capital case?"], prompt_name="query")
scores = (query @ index.T)[0]                                     # cosine similarity
for i in scores.argsort()[::-1][:5]:
    print(round(float(scores[i]), 2), passages[i][:80])
```

Output of `scripts/example_search.py` with the 1.7B model (three passages: Deuteronomy 17:6, Exodus 20:12,
Mishnah Berakhot 1:1, all in vocalised Hebrew):

```
How many witnesses are needed to convict in a capital case?
  0.55  Deuteronomy 17:6        0.07  Exodus 20:12        0.02  Mishnah Berakhot 1:1
מאימתי קורין את שמע בערבית
  0.61  Mishnah Berakhot 1:1    0.08  Exodus 20:12       -0.02  Deuteronomy 17:6
The reward for honoring father and mother
  0.51  Exodus 20:12            0.05  Deuteronomy 17:6    0.04  Mishnah Berakhot 1:1
```

To search your own texts: `python scripts/example_search.py --passages my_texts.txt --query "..."` (one passage per
line). For a large library, encode the passages once on a GPU, store the vectors, and encode only the query at search
time: a search then encodes only the query, about half a second for the 1.7B on a 16-core CPU.

## What it is good for

| you have | you want | instruction (`prompt_name`) | how well it works (KesherBench nDCG@10 where given) |
|---|---|---|---|
| a question, in English or Hebrew | the sources that answer it | `query` | the main use; an English question finds Hebrew/Aramaic passages that have no translation |
| a ruling (e.g. a paragraph of the Mishneh Torah) | its Talmud sources | `sources_of_ruling` | nDCG@10 0.44–0.48 (Hebrew), against 0.34 for the best open model tested |
| a commentary | the verse or passage it explains | `commentary_to_text` | 0.57–0.70 on the Torah and the Talmud, against 0.29–0.47 |
| a passage | its parallels (e.g. Yerushalmi → Bavli) | `parallels` | 0.44–0.48, against 0.33 |
| a passage | its translation | `translation` | 96.7% Recall@1 (1.7B) on Sefaria's benchmark |
| a short topic ("Shabbat candles") | key sources | `topic` | the weak spot: see below |

Real searches with the 1.7B model over an index of 112,528 passages of Tanakh, Mishnah, Bavli and Mishneh Torah
(top results with their cosine similarity):

| query | top results | what happened |
|---|---|---|
| "How many witnesses are needed to convict in a capital case?" | Deuteronomy 17:6 (0.55) · Makkot 5b:22 (0.53) · Makkot 5b:21 (0.52) | the verse, then the Mishnah and Talmud passages that expound it |
| "האם מותר לאכול לפני תפילת שחרית" | Mishneh Torah, Prayer and the Priestly Blessing 6:4 (0.66) · Berakhot 28b:2 (0.56) | a Hebrew question finds the ruling |
| the Hebrew text of Mishneh Torah, Scroll of Esther and Hanukkah 3:2, with `sources_of_ruling`, searching the Bavli | Shabbat 21b:10 (0.52) · Horayot 11b:8 (0.41) | the ruling finds the Talmud's account of Hanukkah, by a wide margin |
| "Shabbat candles", with `topic` | Mishneh Torah, Daily Offerings 3:10 (0.48) · The Chosen Temple 8:12 (0.46) · … · Mishneh Torah, Sabbath 5:1 (0.46, fourth) | a weak case: the Temple lamp comes first, the Sabbath lamp fourth |

**Which model:** the 1.7B is the default (close to the 14B, answers a query in about half a second on a 16-core CPU);
the 0.6B halves vector size and query time for a moderate loss; the 14B is ahead overall, the best of the three on short
topic searches (close to the base model there), and needs a GPU.

**What it is not good for:** short topic searches with the small models (with the 0.6B, add keyword search there), passages or
queries under about 50 characters, general non-Jewish text (use the base model), and decisions of any kind: it finds
sources to check, it does not rule.

## Results (test split)

| model | size | Sefaria bench R@1 (translation matching) | KesherBench score (finding related sources) | English question → Hebrew source | topic → sources, hit@10 (he / en) |
|---|---:|---:|---:|---:|---:|
| **ein-mishpat-ai-0.6b** | 0.6B | 85.7% | 0.521 | 0.460 | 32.0% / 21.2% |
| **ein-mishpat-ai-1.7b** | 1.7B | 96.7% | 0.545 | 0.503 | 40.1% / 29.7% |
| **ein-mishpat-ai-14b** | 14B | 96.8% | 0.564 | 0.515 | 46.9% / 41.1% |
| F2LLM-v2-14B (base of our 14B) | 14B | 93.3% | 0.382 | 0.355 | 49.9% / 39.7% |
| Qwen3-Embedding-8B | 8B | 89.3% | 0.330 | 0.281 | 46.6% / 37.5% † |
| F2LLM-v2-1.7B (base of our 1.7B) | 1.7B | 67.7% | 0.246 | 0.178 | 37.5% / 24.2% |
| F2LLM-v2-0.6B (base of our 0.6B) | 0.6B | 24.7% | 0.175 | 0.103 | 34.5% / 17.4% |
| BGE-M3 | 0.6B | 36.4% | 0.251 | 0.169 | 11.7% / 5.5% † |
| multilingual-e5-large | 0.6B | 19.8% | 0.224 | 0.103 | 25.9% / 0.7% † |
| BM25 (keyword search) | — | — | — | — | 36.0% / — |
| Gemini Embedding 001 (API; published leaderboard #1 on 2026-11-08, not run here) | — | 93.9% (published) | — | — | — |

KesherBench score = mean nDCG@10 over 8 settings; "English question → Hebrew source" = its three English → Hebrew settings; topic → sources = share of Sefaria topic pages with a listed source in the top 10. 95% intervals (over queries) are on the model cards; every model was trained once (one seed). The 14B and the 1.7B are not clearly different on the KesherBench score. † = topic search measured on the pool as it was before the Torah passages were given the Masoretic text (see "The Torah text" below); for the base models measured both ways the change was at most 0.3 points.

A test query keeps all its answer passages, and some lie in works that were used in training as text (never paired with the query). Counting only answers in held-out works, the KesherBench scores are 0.545 / 0.530 / 0.507 (14B / 1.7B / 0.6B) against 0.370 for F2LLM-v2-14B (details on the KesherBench card).

**Before you index:** remove vowel points and cantillation from Hebrew (one line of code, on the model cards); with them, results drop sharply for the small models.

**The Torah text:** KesherBench gives the Torah and the Five Megillot in the Masoretic text. In training, the data pipeline's version rule had picked a modern-Hebrew rendering ("Miqra Mevoar") for these books; the results above are on the Masoretic text.

## How the models were made

All three start from [F2LLM-v2](https://huggingface.co/codefuse-ai) (Apache-2.0, built on Qwen3):

- **0.6B, 1.7B:** full fine-tuning with contrastive learning, one epoch over 2.6M examples. The loss also has a small
  distillation term toward the scores of the 14B teacher; in a short ablation it left the KesherBench score unchanged
  and mainly helped topic search.
- **14B:** LoRA fine-tuning, 500 steps; the adapter's change is scaled by 0.85 and merged, which keeps almost all of the
  gain on finding sources and recovers part of the topic search the full adapter loses.

Training signals, all from existing human-made resources, with no new annotation:
- freely licensed translations
- Sefaria's link graph (commentary → text, Ein Mishpat, Masoret HaShas, quotations, Midrash and Targum → verse)
- LLM-written English translations and search queries (Qwen3.8-27B), used **for training only** and never evaluated

Project rules. Code comments refer to them as "project rule N"; rules 1–4 are enforced by unit tests in `tests/`.
1. **No benchmark leakage:** no benchmark passage (or its ±2 neighbouring segments) and no held-out work ever reaches
   training, in any language or as either side of a link.
2. **Split by work:** held-out sets are whole works, not random segments. A link pair is used for training only if both
   sides are in training works.
3. **Licenses:** only Public Domain, CC0, CC-BY, CC-BY-SA, plus versions that Sefaria lists as "unknown" (used for
   non-commercial purposes only and not redistributed; see the model cards). No non-commercial translations anywhere.
4. **Synthetic text never enters evaluation:** LLM-written text is tagged and kept out of every evaluation set.
5. **Reproduce before trusting:** the evaluation harness reproduces the published Sefaria leaderboard numbers exactly.
6. **Same evaluation for every model:** one harness and one config per benchmark, with fixed seeds. Numbers from
   different harness versions are never compared.
7. **Log every run:** every run is logged with its config, data mix and code hash.
8. **Review before anything public or irreversible.**

## Install

Python 3.10–3.11, PyTorch ≥ 2.3. Tested with PyTorch 2.3, transformers 4.57.6, sentence-transformers 5.7.0 and
peft 0.21.2 (`constraints.txt` holds the torch and numpy pins).

```bash
pip install -e ".[eval]"          # evaluation and using the models
pip install -e ".[train]"         # training as well
python -m pytest -q               # tests (data-dependent ones skip until the data is built)
```

## Reproduce

The project ran on one node with 8 × A100 80GB. The whole project used about 1,500 GPU-hours and 324M LLM tokens.
Long multi-GPU jobs use `torchrun --nproc_per_node 8`. `scripts/run_parallel.py` runs job lists one
job per GPU. Endpoint settings for step 3 go in `.env` (see `.env.example`).

```bash
# 1. data (Sefaria export + API, licenses parsed per version)
python scripts/download_sefaria.py meta && python scripts/download_sefaria.py texts --all
python scripts/download_topics.py
python scripts/download_hf.py                     # base and comparison models
python scripts/build_data.py all                  # versions, segments, links
python scripts/build_m2.py all                    # leakage exclusion, splits, KesherBench, training pairs
python scripts/masoretic.py build && python scripts/masoretic.py apply   # KesherBench: Masoretic text for the Torah

# 2. teacher pass and filters (F2LLM-v2-14B)
python scripts/teacher.py prepare                 # then run_parallel.py on the job lists it writes
python scripts/teacher.py mine --task <task>

# 3. synthetic training data (OpenAI-compatible endpoint; set LLM_BASE_URL, LLM_API_KEY, GEN_MODEL in the environment)
python scripts/translate.py d3a-select && python scripts/translate.py d3a-run
python scripts/translate.py d3b-select && python scripts/translate.py d3b-run
python scripts/filters.py d3a-prepare             # round-trip filter, then d3a-score, d3a-keep, d3a-mine

# 4. training
python scripts/train.py mix configs/train/mix_d3b_all.yaml
torchrun --nproc_per_node 8 scripts/train.py fit configs/train/student_1.7b_full_d3b.yaml
torchrun --nproc_per_node 8 scripts/train.py fit configs/train/student_0.6b_full_d3b.yaml
torchrun --nproc_per_node 8 scripts/train.py fit configs/train/flagship_14b.yaml       # 250 steps
torchrun --nproc_per_node 8 scripts/train.py fit configs/train/flagship_14b_ext.yaml   # +250 steps
python scripts/lora_scale.py merge 85             # the released 14B: adapter scaled by 0.85, then merged
python scripts/lora_scale.py adapter 85           # the adapter repository (lora_alpha 27.2), checked weight by weight

# 5. evaluation
python scripts/eval_sefaria_bench.py <model>      # needs Sefaria/Rabbinic-Hebrew-English-Pairs (evaluation only)
python scripts/kesher_jobs.py <name> <model> --split test   # encode + evaluate job lists for run_parallel.py
python scripts/eval_regression.py --config configs/eval/regression_clean.yaml <model>
python scripts/strict_gold.py                    # test scores counting only answers in held-out works
```

Every run writes its resolved config next to its outputs and logs a code hash over `src/` and `configs/`. Seeds are
fixed. All configs of the final runs are in `configs/train/`.

## Licenses

| part | license |
|---|---|
| code (this repository) | Apache-2.0 |
| models | CC BY-NC 4.0 (base models Apache-2.0) |
| KesherBench | CC BY-SA 4.0; texts keep their own license, given per row |
| texts | from [Sefaria](https://www.sefaria.org); every version is listed with its license in [`ATTRIBUTION.md`](ATTRIBUTION.md) |

## Acknowledgements

Compute resources were provided by the hessian.AI Innovation Lab, funded by the Hessian Ministry for Digital Strategy and Innovation (grant no. S-DIW04/0013/003).

## Citation

```bibtex
@misc{dadashev2026einmishpat,
  author = {Lev Dadashev},
  title  = {ein-mishpat-ai: open embedding models and the KesherBench benchmark for Jewish texts},
  year   = {2026},
  url    = {https://huggingface.co/ai-arie}
}
```
