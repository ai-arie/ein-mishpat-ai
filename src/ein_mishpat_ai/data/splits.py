"""Held-out works (project rule 2 (README): split by work, not by segment).

split_of(title, node_ref) -> "test" | "dev" | None (training-eligible). See configs/splits.yaml.
"""

from __future__ import annotations

import glob
import json
from typing import Any

from ein_mishpat_ai.utils.io import load_yaml, resolve

SPLITS = ("test", "dev")


class Splits:
    def __init__(self, cfg: dict[str, Any] | None = None, schemas_dir: str | None = None):
        cfg = cfg or load_yaml("configs/splits.yaml")
        sdir = schemas_dir or str(resolve("data/raw/sefaria/schemas"))
        self.bases = {s: set(cfg[s]["bases"]) for s in SPLITS}
        self.aliases = {s: set(cfg[s]["node_aliases"]) for s in SPLITS}
        self.work: dict[str, str] = {}  # title -> split, for whole works
        self.multi: dict[str, dict[str, set[str]]] = {}  # title -> split -> node names
        for f in glob.glob(f"{sdir}/*.json"):
            try:
                s = json.load(open(f))
            except ValueError:
                continue
            title = s["title"]
            cats = " > ".join(s.get("categories", []))
            for sp in SPLITS:
                if title in self.bases[sp] or any(
                    cats.startswith(f"Halakhah > Mishneh Torah > {m}")
                    for m in cfg[sp]["mishneh_torah_sefarim"]
                ):
                    self.work[title] = sp
        mt_books = {t: sp for t, sp in self.work.items() if t.startswith("Mishneh Torah, ")}
        for sp in SPLITS:
            self.bases[sp] |= {t for t, x in mt_books.items() if x == sp}
        schemas = []
        for f in glob.glob(f"{sdir}/*.json"):
            try:
                schemas.append(json.load(open(f)))
            except ValueError:
                continue
        known = {s["title"] for s in schemas}
        for s in schemas:
            title = s["title"]
            if title in self.work:
                continue
            # The title wins over base_text_titles ("Chidushei Agadot on Keritot" lists Chullin).
            decided, sp = self._title_rule(title, known)
            if decided:
                if sp:
                    self.work[title] = sp
                continue
            bt = {b["en"] for b in s.get("base_text_titles") or [] if isinstance(b, dict)}
            if not bt:
                continue
            for sp in SPLITS:
                held = bt & self.bases[sp]
                if held and held == bt:
                    self.work[title] = sp
                elif held:
                    self.multi.setdefault(title, {})[sp] = held | self.aliases[sp]

        self.known = known

    def _title_rule(self, title: str, known: set[str]) -> tuple[bool, str | None]:
        """(decided, split) from the title alone: "X on <work>" or "…, <book>"."""
        _, sep, tail = title.rpartition(" on ")
        if sep and tail in known:
            return True, next((sp for sp in SPLITS if tail in self.bases[sp]), None)
        last = title.rsplit(", ", 1)[-1] if ", " in title else ""
        hit = [sp for sp in SPLITS if last in self.bases[sp] | self.aliases[sp]]
        return (True, hit[0]) if hit else (False, None)

    def split_of(self, title: str, node_ref: str = "") -> str | None:
        sp = self.work.get(title)
        if sp:
            return sp
        if title not in self.known:  # no schema: title rules only
            return self._title_rule(title, self.known)[1]
        by = self.multi.get(title)
        if by and node_ref:
            parts = set(node_ref[len(title) :].split(", ")) if node_ref.startswith(title) else set()
            for s, names in by.items():
                if parts & names:
                    return s
        return None

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for sp in SPLITS:
            out[sp] = sorted(t for t, x in self.work.items() if x == sp)
            out[f"{sp}_multi"] = sorted(t for t, by in self.multi.items() if sp in by)
        return out
