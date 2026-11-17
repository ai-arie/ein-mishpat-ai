"""Flag versions that are machine- or LLM-translated (project rule 4 (README): never in evaluation).

Matches the version title, source and versionNotes. Flagged versions get `synthetic = true` in the
versions table and segments; every eval set filters them out. The match list is written to
data/meta/synthetic_versions.tsv for review.
"""

from __future__ import annotations

import re

_CI = re.compile(
    r"deepl|google[ -]?translat|translate\.google|machine[- ]translat|chat ?gpt|\bgpt-?\d|\bllm\b"
    r"|artificial intelligence|\bai[- ](assisted|generated|translat|powered)|\(claude"
    r"|claude[ -](opus|sonnet|haiku|\d)|anthropic|gemini[ -](pro|flash|\d)|openai"
    r"|automat(ed|ic|ically)[ -]translat|computer[- ]translat",
    re.I,
)
_CS = re.compile(r"\bAI\b")


def is_synthetic(*fields: str | None) -> str:
    """The matched marker, or '' if none."""
    for f in fields:
        if not f:
            continue
        m = _CI.search(f) or _CS.search(f)
        if m:
            return m.group(0)
    return ""
