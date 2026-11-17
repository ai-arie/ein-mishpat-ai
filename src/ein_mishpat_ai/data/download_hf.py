"""Base models and the leaderboard table from Hugging Face (setup notes S3.2).

Does NOT download Sefaria/Rabbinic-Hebrew-English-Pairs (needs explicit approval).
"""

from __future__ import annotations

import shutil
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from ein_mishpat_ai.utils.io import resolve
from ein_mishpat_ai.utils.log import get_logger

log = get_logger("download_hf", "download_hf.log")

FORBIDDEN = {"Sefaria/Rabbinic-Hebrew-English-Pairs"}


def repo_size(repo_id: str, repo_type: str = "model") -> int:
    info = HfApi().repo_info(repo_id, repo_type=repo_type, files_metadata=True)
    return sum((f.size or 0) for f in info.siblings or [])


def run_hf(cfg: dict[str, Any]) -> None:
    hf = cfg["hf"]
    for item in hf["dataset_files"]:
        if item["repo"] in FORBIDDEN:
            raise ValueError(f"{item['repo']} needs explicit approval")
        src = hf_hub_download(item["repo"], item["file"], repo_type="dataset")
        dest = resolve(item["out"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        log.info("dataset file %s/%s -> %s", item["repo"], item["file"], dest)
    for repo in hf["models"]:
        size = repo_size(repo)
        log.info("model %s: %.1f GB, downloading", repo, size / 1e9)
        path = snapshot_download(repo, max_workers=8)
        log.info("model %s done: %s", repo, path)
