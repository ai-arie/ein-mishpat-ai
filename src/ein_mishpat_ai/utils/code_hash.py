"""Code version without git: a hash over src/ and configs/."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ein_mishpat_ai.utils.io import project_root

_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".json"}


def code_hash(root: Path | None = None) -> str:
    """sha256 (first 16 hex chars) over all source and config files, in sorted path order."""
    root = root or project_root()
    h = hashlib.sha256()
    for sub in ("src", "configs"):
        for p in sorted((root / sub).rglob("*")):
            if p.is_file() and p.suffix in _SUFFIXES and "__pycache__" not in p.parts:
                h.update(str(p.relative_to(root)).encode())
                h.update(b"\0")
                h.update(p.read_bytes())
                h.update(b"\0")
    return h.hexdigest()[:16]
