"""Versions table and segments table.

data/meta/versions.parquet        one row per kept version (from the download manifest)
data/processed/segments/*.parquet one part per version (resumable), concatenated into
data/processed/segments.parquet   ref, title, node_ref, address, version_id, lang, text_* …
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ein_mishpat_ai.data.licenses import is_allowed, restrictive_editions
from ein_mishpat_ai.data.normalize import clean, plain_hebrew
from ein_mishpat_ai.data.refs import walk_version
from ein_mishpat_ai.data.synthetic import is_synthetic
from ein_mishpat_ai.utils.io import read_jsonl, resolve
from ein_mishpat_ai.utils.log import get_logger

log = get_logger("segments", "build_data.log")

LANG = {"Hebrew": "he", "English": "en"}
TOKENIZER = "codefuse-ai/F2LLM-v2-0.6B"  # Qwen3 tokenizer, shared by all our base models

SEGMENT_SCHEMA = pa.schema(
    [
        ("ref", pa.string()),
        ("title", pa.string()),
        ("node_ref", pa.string()),
        ("address", pa.list_(pa.int32())),
        ("seg_order", pa.int32()),
        ("version_id", pa.string()),
        ("lang", pa.string()),
        ("actual_language", pa.string()),
        ("category", pa.string()),
        ("categories", pa.string()),
        ("license", pa.string()),
        ("text_raw", pa.string()),
        ("text_clean", pa.string()),
        ("text_plain", pa.string()),
        ("n_chars", pa.int32()),
        ("n_words", pa.int32()),
        ("n_tokens", pa.int32()),
    ]
)


def version_id(title: str, language: str, version_title: str) -> str:
    # exact title: some distinct versions differ only by surrounding whitespace
    key = f"{title}\x1f{language}\x1f{version_title}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def schema_file(sefaria_dir: Path, title: str) -> Path:
    return sefaria_dir / "schemas" / (title.replace(" ", "_") + ".json")


def build_versions(cfg: dict[str, Any]) -> pa.Table:
    """Kept versions from the manifest (last record per URL)."""
    sdir = resolve(cfg["sefaria"]["out_dir"])
    last: dict[str, dict[str, Any]] = {}
    for r in read_jsonl(sdir / "versions_manifest.jsonl"):
        last[r["url"]] = r
    rows = []
    inherit = cfg["licenses"].get("unknown_inherits_restrictive_edition", False)
    restrictive = restrictive_editions(list(last.values())) if inherit else {}
    excluded = []
    for r in last.values():
        if r["status"] != "kept":
            continue
        if not is_allowed(r["license_norm"]):
            raise ValueError(f"kept version with disallowed license: {r['url']}")
        edition = (r["versionTitle"].strip(), r["language"])
        if r["license_norm"] == "unknown" and r.get("license_source") != "override":
            if edition in restrictive:
                excluded.append(
                    (r["title"], r["language"], r["versionTitle"], restrictive[edition])
                )
                continue
        rows.append(
            {
                "version_id": version_id(r["title"], r["language"], r["versionTitle"]),
                "title": r["title"],
                "language": r["language"],
                "lang": LANG[r["language"]],
                "actual_language": r.get("actualLanguage") or "",
                "version_title": r["versionTitle"].strip(),
                "version_source": r.get("versionSource") or "",
                "license": r["license_norm"],
                "license_raw": str(r.get("license_raw")),
                "license_source": r.get("license_source") or "",
                "is_source": bool(r.get("isSource")),
                "category": r["categories"][0] if r["categories"] else "",
                "categories": " > ".join(r["categories"]),
                "path": r["path"],
                "sha256": r["sha256"],
            }
        )
    flagged = []
    for x in rows:
        notes = json.loads((sdir / x["path"]).read_bytes()).get("versionNotes") or ""
        x["synthetic_marker"] = is_synthetic(x["version_title"], x["version_source"], notes)
        x["synthetic"] = bool(x["synthetic_marker"])
        if x["synthetic"]:
            flagged.append((x["title"], x["language"], x["version_title"], x["synthetic_marker"]))
    rows.sort(key=lambda x: (x["title"], x["language"], x["version_title"]))
    ids = [x["version_id"] for x in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("version_id collision")
    out = resolve(cfg["processed"]["versions"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out, compression="zstd")
    with (out.parent / "excluded_unknown_editions.tsv").open("w") as f:
        f.write("title\tlanguage\tversionTitle\tedition_license_elsewhere\n")
        for e in sorted(excluded):
            f.write("\t".join(e) + "\n")
    log.info(
        "versions: %d unknown-license versions excluded (edition restrictive elsewhere)",
        len(excluded),
    )
    with (out.parent / "synthetic_versions.tsv").open("w") as f:
        f.write("title\tlanguage\tversionTitle\tmarker\n")
        for e in sorted(flagged):
            f.write("\t".join(e) + "\n")
    log.info("versions: %d flagged synthetic (machine/LLM translation)", len(flagged))
    log.info("versions: %d kept versions -> %s", len(rows), out)
    return table


_TOK = None


def _tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained(TOKENIZER)
    return _TOK


def flatten_version(sefaria_dir: str, v: dict[str, Any], out_path: str) -> tuple[str, int, str]:
    """One version -> parquet part. Returns (version_id, n_segments, problem or '')."""
    sdir = Path(sefaria_dir)
    doc = json.loads((sdir / v["path"]).read_bytes())
    sfile = schema_file(sdir, v["title"])
    if not sfile.exists() or sfile.stat().st_size == 0:
        return v["version_id"], 0, "no_schema"
    schema = json.loads(sfile.read_bytes())["schema"]
    rows: dict[str, list[Any]] = {f.name: [] for f in SEGMENT_SCHEMA}
    he = v["lang"] == "he"
    seen: set[str] = set()
    for order, (node_ref, ref, addr, raw) in enumerate(
        walk_version(v["title"], schema, doc["text"])
    ):
        c = clean(raw)
        if not c or ref in seen:  # a few schemas have two sibling nodes with one title
            continue
        seen.add(ref)
        p = plain_hebrew(c) if he else None
        rows["ref"].append(ref)
        rows["title"].append(v["title"])
        rows["node_ref"].append(node_ref)
        rows["address"].append(list(addr))
        rows["seg_order"].append(order)
        rows["version_id"].append(v["version_id"])
        rows["lang"].append(v["lang"])
        rows["actual_language"].append(v["actual_language"])
        rows["category"].append(v["category"])
        rows["categories"].append(v["categories"])
        rows["license"].append(v["license"])
        rows["text_raw"].append(raw)
        rows["text_clean"].append(c)
        rows["text_plain"].append(p)
        rows["n_chars"].append(len(c))
        rows["n_words"].append(len((p or c).split()))
    n = len(rows["ref"])
    problem = "" if n else "no_segments"
    if n:
        enc = _tokenizer()(rows["text_clean"], add_special_tokens=False)["input_ids"]
        rows["n_tokens"] = [len(x) for x in enc]
    else:
        rows["n_tokens"] = []
    tmp = Path(out_path + ".part")
    pq.write_table(pa.table(rows, schema=SEGMENT_SCHEMA), tmp, compression="zstd")
    tmp.replace(out_path)
    return v["version_id"], n, problem


def build_segments(cfg: dict[str, Any], workers: int = 48) -> None:
    sdir = resolve(cfg["sefaria"]["out_dir"])
    versions = pq.read_table(resolve(cfg["processed"]["versions"])).to_pylist()
    parts = resolve(cfg["processed"]["segment_parts"])
    parts.mkdir(parents=True, exist_ok=True)
    todo = [v for v in versions if not (parts / f"{v['version_id']}.parquet").exists()]
    log.info("segments: %d versions, %d to flatten", len(versions), len(todo))
    problems: dict[str, str] = {}
    with ProcessPoolExecutor(workers) as ex:
        futs = {
            ex.submit(flatten_version, str(sdir), v, str(parts / f"{v['version_id']}.parquet")): v
            for v in todo
        }
        for i, fut in enumerate(as_completed(futs), 1):
            vid, _n, problem = fut.result()
            if problem:
                problems[vid] = problem
            if i % 1000 == 0:
                log.info("flattened %d/%d", i, len(todo))
    if problems:
        (parts / "_problems.json").write_text(json.dumps(problems, indent=1))
        log.info("versions with problems: %d (see _problems.json)", len(problems))
    tables = []
    for v in versions:
        t = pq.read_table(parts / f"{v['version_id']}.parquet")
        flag = pa.array([v["synthetic"]] * t.num_rows, pa.bool_())
        tables.append(t.append_column("synthetic", flag))
    table = pa.concat_tables(tables)
    out = resolve(cfg["processed"]["segments"])
    pq.write_table(table, out, compression="zstd", row_group_size=200_000)
    log.info("segments: %d rows -> %s", table.num_rows, out)
