"""Text normalisation.

text_clean: footnotes and HTML removed, entities unescaped, whitespace collapsed. Bold/plain parts
            of Steinsaltz-style texts are both kept (tags dropped, text kept).
text_plain: Hebrew/Aramaic only — text_clean without cantillation and niqqud, maqaf -> space,
            paseq and Tanakh parasha markers ({פ}, {ס}) removed; geresh/gershayim kept.
"""

from __future__ import annotations

import html
import re

_FOOTNOTE_MARKER = re.compile(r"<sup\b[^>]*>.*?</sup>", re.S | re.I)
_FOOTNOTE_OPEN = re.compile(r'<i\b[^>]*class="[^"]*footnote[^"]*"[^>]*>', re.I)
_I_TAG = re.compile(r"<(/?)i\b[^>]*>", re.I)
_BR = re.compile(r"<br\s*/?>", re.I)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")

# U+0591–U+05AF cantillation; U+05B0–U+05BD, U+05BF, U+05C1–U+05C2, U+05C4–U+05C5, U+05C7 points.
_MARKS = re.compile("[֑-ְ֯-ׇֽֿׁׂׅׄ]")
_PASEQ = "׀"
_MAQAF = "־"
_PARASHA = re.compile(r"\{[פס]\}")


def _strip_footnotes(s: str) -> str:
    """Remove <i class="footnote">…</i> including nested <i> tags."""
    while True:
        m = _FOOTNOTE_OPEN.search(s)
        if not m:
            return s
        depth, pos = 1, m.end()
        for t in _I_TAG.finditer(s, pos):
            depth += -1 if t.group(1) else 1
            if depth == 0:
                s = s[: m.start()] + " " + s[t.end() :]
                break
        else:  # unbalanced: drop to the end
            return s[: m.start()]


def clean(raw: str) -> str:
    s = _FOOTNOTE_MARKER.sub("", raw)
    s = _strip_footnotes(s)
    s = _BR.sub(" ", s)
    s = _TAG.sub("", s)
    s = html.unescape(s).replace(" ", " ")
    return _WS.sub(" ", s).strip()


def plain_hebrew(cleaned: str) -> str:
    s = _MARKS.sub("", cleaned)
    s = s.replace(_MAQAF, " ").replace(_PASEQ, " ")
    s = _PARASHA.sub(" ", s)
    return _WS.sub(" ", s).strip()
