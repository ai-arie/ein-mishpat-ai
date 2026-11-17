import json
import re
from pathlib import Path

import pytest

from ein_mishpat_ai.data.licenses import (
    ALLOWED,
    LICENSE_VALUES,
    find_override,
    is_allowed,
    normalize_license,
)
from ein_mishpat_ai.utils.io import load_yaml, project_root, read_jsonl

CASES = {
    # allowed
    "Public Domain": "public_domain",
    "public domain": "public_domain",
    "PD": "public_domain",
    "CC0": "cc0",
    "CC0 1.0": "cc0",
    "CC-BY": "cc_by",
    "CC BY": "cc_by",
    "cc-by": "cc_by",
    "CC-BY 3.0": "cc_by",
    "CC-BY 4.0": "cc_by",
    "CC_BY_4.0": "cc_by",
    "Creative Commons Attribution 4.0 International License": "cc_by",
    "CC-BY-SA": "cc_by_sa",
    "CC-BY-SA 3.0": "cc_by_sa",
    "CC BY-SA 4.0": "cc_by_sa",
    # excluded
    "CC-BY-NC": "cc_by_nc",
    "CC-BY-NC 4.0": "cc_by_nc",
    "CC BY-NC": "cc_by_nc",
    "CC-BY-NC-SA": "cc_by_nc_sa",
    "CC-BY-NC-SA 4.0": "cc_by_nc_sa",
    "CC BY-NC-SA 3.0": "cc_by_nc_sa",
    "CC-BY-NC-ND": "cc_by_nc",
    "CC-BY-ND": "cc_by_nd",
    "Creative Commons Attribution-NonCommercial-ShareAlike 4.0": "cc_by_nc_sa",
    "Copyright": "copyright",
    "Copyright 2015 Koren": "copyright",
    "": "unknown",
    "   ": "unknown",
    "unknown": "unknown",
    "Unknown": "unknown",
    "Other": "unrecognized",
    "CC": "unrecognized",
    "Public Domain?": "unrecognized",
    "All rights reserved": "unrecognized",
}


@pytest.mark.parametrize("raw,expected", sorted(CASES.items()))
def test_normalize(raw, expected):
    assert normalize_license(raw) == expected


def test_missing_is_unknown():
    assert normalize_license(None) == "unknown"


@pytest.mark.parametrize("raw", [0, [], {}])
def test_non_string_is_unrecognized(raw):
    assert normalize_license(raw) == "unrecognized"
    assert not is_allowed(normalize_license(raw))


def test_allowed_set():
    assert {v for v in LICENSE_VALUES if is_allowed(v)} == {
        "public_domain",
        "cc0",
        "cc_by",
        "cc_by_sa",
        "unknown",
    }


def test_config_allowed_matches_code():
    cfg = load_yaml("configs/data.yaml")
    assert set(cfg["licenses"]["allowed"]) == set(ALLOWED)


OV = [{"title": "T", "versionTitle": "V", "language": "Hebrew", "license": "public_domain"}]


def test_override_exact_match_lifts_unknown():
    assert find_override(OV, "T", "V ", "Hebrew", "unknown") is OV[0]
    assert find_override(OV, "T", "V", "Hebrew", None) is OV[0]
    assert find_override(OV, "T", "V", "English", "unknown") is None
    assert find_override(OV, "T", "Other", "Hebrew", "unknown") is None


@pytest.mark.parametrize("raw", ["CC-BY-NC", "CC-BY-NC-SA", "CC-BY-ND", "Copyright", "CC-BY"])
def test_override_never_replaces_a_known_license(raw):
    assert find_override(OV, "T", "V", "Hebrew", raw) is None


def test_override_to_disallowed_license_fails():
    bad = [dict(OV[0], license="cc_by_nc")]
    with pytest.raises(ValueError):
        find_override(bad, "T", "V", "Hebrew", "unknown")


def test_configured_overrides_are_allowed():
    for o in load_yaml("configs/data.yaml")["licenses"].get("overrides") or []:
        assert is_allowed(o["license"]), o


def test_nc_never_allowed():
    for raw in CASES:
        if "nc" in raw.lower().replace("-", " ").split() or "noncommercial" in raw.lower():
            assert not is_allowed(normalize_license(raw)), raw


def _sefaria_dir() -> Path:
    return project_root() / "data" / "raw" / "sefaria"


def test_seen_license_strings_never_allow_restricted():
    """Every raw string actually seen in the manifest: NC / ND / copyright never map to allowed."""
    manifest = _sefaria_dir() / "versions_manifest.jsonl"
    if not manifest.exists():
        pytest.skip("no manifest yet")
    restricted = {
        "nc",
        "nd",
        "noncommercial",
        "commercial",
        "noderivatives",
        "derivatives",
        "noderivs",
        "copyright",
    }
    seen = {
        r.get("license_raw") for r in read_jsonl(manifest) if isinstance(r.get("license_raw"), str)
    }
    for raw in seen:
        tokens = set(re.findall(r"[a-z]+", raw.lower()))
        if tokens & restricted:
            assert not is_allowed(normalize_license(raw)), raw


def test_no_disallowed_files_on_disk():
    """Every version file on disk has an allowed license (re-read from the file, or API-resolved)."""
    root = _sefaria_dir()
    json_dir = root / "json"
    if not json_dir.exists():
        pytest.skip("no texts downloaded yet")
    api_resolved = {}
    for r in read_jsonl(root / "versions_manifest.jsonl"):
        if r.get("status") == "kept" and r.get("path"):
            if r.get("license_source") == "api":
                api_resolved[r["path"]] = r["license_raw"]
            elif r.get("license_source") == "override":
                api_resolved[r["path"]] = r["license_norm"]
    bad = []
    for f in json_dir.rglob("*.json"):
        assert f.name != "merged.json", f
        rel = str(f.relative_to(root))
        lic = json.loads(f.read_bytes()).get("license")
        if is_allowed(normalize_license(lic)):
            continue
        if rel in api_resolved and (
            is_allowed(api_resolved[rel]) or is_allowed(normalize_license(api_resolved[rel]))
        ):
            continue
        bad.append((rel, lic))
    assert not bad, bad[:20]


@pytest.mark.parametrize(
    "title,tag",
    [
        ("Tora ve Aftara [tr]", "tr"),
        ("Biblia de Ferrara [lad]", "lad"),
        ("Yiddish Translation [YI]", "yi"),
        ("Sefaria Community Translation", None),
        ("The Contemporary Torah [en]", "en"),
        ("Rashi [on] Genesis, 1900", None),
    ],
)
def test_title_language_tag(title, tag):
    from ein_mishpat_ai.data.download_sefaria import title_language_tag

    assert title_language_tag(title) == tag


def test_no_tagged_foreign_translation_kept_as_english():
    from ein_mishpat_ai.data.download_sefaria import title_language_tag

    manifest = _sefaria_dir() / "versions_manifest.jsonl"
    if not manifest.exists():
        pytest.skip("no manifest yet")
    last = {r["url"]: r for r in read_jsonl(manifest)}
    bad = [
        r["versionTitle"]
        for r in last.values()
        if r["status"] == "kept"
        and r["language"] == "English"
        and title_language_tag(r.get("versionTitle", "")) not in (None, "en")
    ]
    assert not bad, bad
