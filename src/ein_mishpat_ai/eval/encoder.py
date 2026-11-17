"""Sentence-transformers encoder with explicit prompt and length handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ein_mishpat_ai.utils.io import load_yaml


@dataclass
class EncoderSpec:
    name: str
    hf_id: str
    prompt_style: str = "none"  # "none" | "instruct" | "fixed" (query_prompt / doc_prompt, e.g. e5)
    query_prompt: str = ""
    doc_prompt: str = (
        ""  # applied by the corpus jobs (scripts/kesher_jobs.py → encode_corpus --prefix)
    )
    max_length: int = 512
    batch_size: int = 32
    dtype: str = "auto"


def model_spec(name: str, **overrides: Any) -> EncoderSpec:
    entry = load_yaml("configs/eval/models.yaml")["models"][name]
    return EncoderSpec(name=name, **{**entry, **overrides})


class Encoder:
    def __init__(self, spec: EncoderSpec, device: str = "cuda"):
        import torch
        from sentence_transformers import SentenceTransformer

        dtype = spec.dtype if spec.dtype == "auto" else getattr(torch, spec.dtype)
        self.spec = spec
        self.model = SentenceTransformer(
            spec.hf_id, device=device, model_kwargs={"torch_dtype": dtype}
        )
        self.model.max_seq_length = min(spec.max_length, self.model.max_seq_length)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def query_prefix(self, task: str | None) -> str:
        if self.spec.prompt_style == "instruct" and task:
            return f"Instruct: {task}\nQuery: "
        if self.spec.prompt_style == "fixed":
            return self.spec.query_prompt
        return ""

    def encode(self, texts: list[str], prefix: str = "", chunked: bool = False) -> np.ndarray:
        """L2-normalised float32 embeddings. `chunked` mimics the Space: one encode call per batch."""
        texts = [prefix + t for t in texts] if prefix else list(texts)
        bs = self.spec.batch_size
        kw = dict(batch_size=bs, normalize_embeddings=True, convert_to_numpy=True)
        if chunked:
            parts = [
                self.model.encode(texts[i : i + bs], show_progress_bar=False, **kw)
                for i in range(0, len(texts), bs)
            ]
            return np.vstack(parts).astype(np.float32)
        return self.model.encode(texts, show_progress_bar=False, **kw).astype(np.float32)
