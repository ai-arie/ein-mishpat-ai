"""BM25 baseline (bm25s). Hebrew: one leading prefix letter (ו ה ב ל מ ש כ) stripped from tokens
longer than 3 letters, repeated up to twice; English: lowercase. Same tokenizer for corpus and
queries."""

from __future__ import annotations

import re

import numpy as np

_TOKEN = re.compile(r"[\wא-ת׳״\"']+")
_PREFIX = set("והבלמשכ")


def tokenize(text: str, lang: str) -> list[str]:
    toks = _TOKEN.findall(text.lower() if lang == "en" else text)
    if lang != "he":
        return toks
    out = []
    for t in toks:
        t = t.strip("\"'׳״")
        for _ in range(2):
            if len(t) > 3 and t[0] in _PREFIX:
                t = t[1:]
        if t:
            out.append(t)
    return out


def bm25_topk(corpus: list[str], queries: list[str], lang: str, k: int) -> np.ndarray:
    import bm25s

    retriever = bm25s.BM25()
    retriever.index([tokenize(t, lang) for t in corpus], show_progress=False)
    q = [tokenize(t, lang) or ["\u0000"] for t in queries]
    idx, _ = retriever.retrieve(q, k=min(k, len(corpus)), show_progress=False, n_threads=16)
    return np.asarray(idx)
