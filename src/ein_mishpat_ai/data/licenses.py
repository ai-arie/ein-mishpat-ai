"""License normalisation. Allowed: public_domain, cc0, cc_by, cc_by_sa, unknown (project rule 3 (README)).

`unknown` means Sefaria's own label is the literal string "unknown" or the field is missing; Sefaria
believes these texts are public domain (Sefaria's reply, 2026-11). Any other string we cannot parse
becomes `unrecognized`, which is excluded, so a new restrictive wording is never let in by accident.
`licenses.overrides` (configs/data.yaml) records versions we treat as public domain although Sefaria
says unknown; a restrictive license (NC / ND / copyright) is never overridden.
"""

from __future__ import annotations

import re
from typing import Any

ALLOWED = frozenset({"public_domain", "cc0", "cc_by", "cc_by_sa", "unknown"})

LICENSE_VALUES = (
    "public_domain",
    "cc0",
    "cc_by",
    "cc_by_sa",
    "cc_by_nc",
    "cc_by_nc_sa",
    "cc_by_nd",
    "copyright",
    "unknown",
    "unrecognized",
)

# Long-form words → short CC tokens, applied before separators are removed.
_WORDS = (
    ("creative commons", "cc"),
    ("creativecommons", "cc"),
    ("attribution", "by"),
    ("non-commercial", "nc"),
    ("noncommercial", "nc"),
    ("non commercial", "nc"),
    ("share-alike", "sa"),
    ("sharealike", "sa"),
    ("share alike", "sa"),
    ("no-derivatives", "nd"),
    ("noderivatives", "nd"),
    ("no derivatives", "nd"),
    ("noderivs", "nd"),
)

# Collapsed CC strings (after "cc", version numbers removed) → normalised value.
_CC = {
    "by": "cc_by",
    "bysa": "cc_by_sa",
    "bync": "cc_by_nc",
    "byncsa": "cc_by_nc_sa",
    "bysanc": "cc_by_nc_sa",
    "byncnd": "cc_by_nc",
    "bynd": "cc_by_nd",
}


def normalize_license(raw: object) -> str:
    """Map a raw license string from Sefaria to one of LICENSE_VALUES."""
    if raw is None:
        return "unknown"
    if not isinstance(raw, str):
        return "unrecognized"
    s = raw.strip().lower()
    if not s or s == "unknown":
        return "unknown"
    if s.startswith("copyright") or s.startswith("©") or s.startswith("(c)"):
        return "copyright"
    for long, short in _WORDS:
        s = s.replace(long, short)
    collapsed = re.sub(r"[\s\-_.,/()]+", "", s)
    if collapsed in {"publicdomain", "pd"}:
        return "public_domain"
    if re.fullmatch(r"cc0(10)?", collapsed) or collapsed in {"cczero", "publicdomaincc0"}:
        return "cc0"
    if collapsed.startswith("cc"):
        rest = re.sub(r"\d+", "", collapsed[2:])  # drop version numbers like 3.0 / 4.0
        rest = rest.removesuffix("license").removesuffix("international")
        if rest in _CC:
            return _CC[rest]
        # Unrecognised CC variant: never guess "allowed". Classify by its restrictive parts.
        if "nc" in rest:
            return "cc_by_nc_sa" if "sa" in rest else "cc_by_nc"
        if "sa" in rest:
            return "cc_by_sa"
        if "nd" in rest:
            return "cc_by_nd"
        return "unrecognized"
    return "unrecognized"


def is_allowed(norm: str) -> bool:
    return norm in ALLOWED


def find_override(
    overrides: list[dict[str, Any]], title: str, version_title: str, language: str, raw: object
) -> dict[str, Any] | None:
    """The override entry for this exact version, or None. Applies only while Sefaria says unknown."""
    if normalize_license(raw) != "unknown":
        return None
    for o in overrides:
        if (
            o["title"] == title
            and o["versionTitle"].strip() == (version_title or "").strip()
            and o["language"] == language
        ):
            if not is_allowed(o["license"]):
                raise ValueError(f"override to a disallowed license: {o}")
            return o
    return None


RESTRICTIVE = frozenset({"cc_by_nc", "cc_by_nc_sa", "cc_by_nd", "copyright"})


def restrictive_editions(records: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    """(versionTitle, language) -> a restrictive license that the same edition carries elsewhere.

    Used to exclude "unknown" versions of an edition that Sefaria labels NC / ND / copyright for
    other works (e.g. two tractates of the William Davidson Edition labelled unknown)."""
    out: dict[tuple[str, str], str] = {}
    for r in records:
        norm = r.get("license_norm")
        if norm in RESTRICTIVE:
            out.setdefault(((r.get("versionTitle") or "").strip(), r["language"]), norm)
    return out
