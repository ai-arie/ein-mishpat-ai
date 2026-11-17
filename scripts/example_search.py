"""Smallest complete example: index a few passages and search them by meaning, in Hebrew or English.

usage: python scripts/example_search.py
       python scripts/example_search.py --query "The reward for honoring father and mother"
       python scripts/example_search.py --passages my_texts.txt --query "..." --prompt topic
       python scripts/example_search.py --model ai-arie/ein-mishpat-ai-0.6b

--passages: a text file with one passage per line (Hebrew/Aramaic or English). Without it, three short passages are
used. Needs only `pip install sentence-transformers`. A retrieval aid, not a halachic authority.
"""

from __future__ import annotations

import argparse
import re

from sentence_transformers import SentenceTransformer

PASSAGES = {  # with vowel points and cantillation, as many digital editions have them
    "Deuteronomy 17:6": "עַל־פִּי שְׁנַיִם עֵדִים אוֹ שְׁלֹשָׁה עֵדִים יוּמַת הַמֵּת לֹא יוּמַת עַל־פִּי עֵד אֶחָד׃",
    "Exodus 20:12": "כַּבֵּד אֶת־אָבִיךָ וְאֶת־אִמֶּךָ לְמַעַן יַאֲרִכוּן יָמֶיךָ עַל הָאֲדָמָה אֲשֶׁר־יְהוָה אֱלֹהֶיךָ נֹתֵן לָךְ׃",
    "Mishnah Berakhot 1:1": "מֵאֵמָּתַי קוֹרִין אֶת שְׁמַע בַּעֲרָבִים? מִשָּׁעָה שֶׁהַכֹּהֲנִים נִכְנָסִים לֹאכַל בִּתְרוּמָתָן, עַד סוֹף "
                            "הָאַשְׁמֹרֶת הָרִאשׁוֹנָה.",
}
QUERIES = ["How many witnesses are needed to convict in a capital case?", "מאימתי קורין את שמע בערבית",
           "The reward for honoring father and mother"]


def strip_marks(text: str) -> str:
    """Remove vowel points and cantillation. The models were trained on Hebrew without them; keep this step."""
    text = re.sub("[֑-ְ֯-ׇֽֿׁׂׅׄ]", "", text)
    return " ".join(text.replace("־", " ").replace("׀", " ").split())  # maqaf, paseq -> space


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="ai-arie/ein-mishpat-ai-1.7b", help="model name on Hugging Face or a local folder")
    ap.add_argument("--passages", help="text file, one passage per line")
    ap.add_argument("--query", action="append", help="a question, a topic or a passage (may be given several times)")
    ap.add_argument("--prompt", default="query", help="instruction for the query: query, topic, sources_of_ruling, "
                                                      "commentary_to_text, parallels, translation, … (see the model card)")
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()

    if a.passages:
        lines = [x.strip() for x in open(a.passages, encoding="utf-8") if x.strip()]
        passages = {f"line {i + 1}": x for i, x in enumerate(lines)}
    else:
        passages = PASSAGES
    model = SentenceTransformer(a.model)
    names = list(passages)
    index = model.encode([strip_marks(passages[n]) for n in names])           # passages: no instruction
    for q in a.query or QUERIES:
        vec = model.encode([strip_marks(q)], prompt_name=a.prompt)            # queries: an instruction
        scores = (vec @ index.T)[0]                                           # cosine similarity (vectors are normalised)
        print(f"\n{q}")
        for i in scores.argsort()[::-1][: a.top]:
            print(f"  {scores[i]:.2f}  {names[i]}: {passages[names[i]][:80]}")


if __name__ == "__main__":
    main()
