"""OpenAI-compatible chat client for synthetic data, standard library only.

Endpoint from the environment or the project's .env (written by hand, mode 600, never
committed): LLM_BASE_URL, LLM_API_KEY, GEN_MODEL. The key is never logged or stored.
Every response is cached in a jsonl file keyed by sha1(model, messages, params), so reruns resume
and every output stays traceable to its exact prompt and sampling settings. Each record carries its
`task` and `time` (from 2026-11-03) for the token accounting.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ein_mishpat_ai.utils.io import resolve
from ein_mishpat_ai.utils.log import get_logger

log = get_logger("llm", "gen.log")
ENV_KEYS = ("LLM_BASE_URL", "LLM_API_KEY", "GEN_MODEL")
THINK = re.compile(r"<think>.*?</think>", re.S)


def load_env(path: str | Path = ".env") -> None:
    p = resolve(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            if k in ENV_KEYS and k not in os.environ:
                os.environ[k] = v.strip().strip("'\"")


def request_key(model: str, messages: list[dict[str, str]], params: dict[str, Any]) -> str:
    blob = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()


class LLM:
    def __init__(
        self, cache: str | Path, concurrency: int = 32, retries: int = 6, task: str = "untagged"
    ):
        load_env()
        missing = [k for k in ENV_KEYS if not os.environ.get(k)]
        if missing:
            raise SystemExit(f"missing in environment/.env: {missing}")
        self.base = os.environ["LLM_BASE_URL"].rstrip("/")
        self.model = os.environ["GEN_MODEL"]
        self._key = os.environ["LLM_API_KEY"]
        self.concurrency = concurrency
        self.task = task
        self.retries = retries
        self.cache_path = resolve(cache)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict[str, Any]] = {}
        readable = [self.cache_path]
        if os.environ.get("EIN_SHELL") == "helper":
            # appends from two machines could interleave: the second shell has its own file, which
            # the token accounting counts too; it still reuses answers from the main file
            self.cache_path = self.cache_path.with_name(self.cache_path.stem + "_helper.jsonl")
            readable.append(self.cache_path)
        for path in readable:
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                try:
                    r = json.loads(line)
                    self.cache[r["key"]] = r
                except (ValueError, KeyError):
                    continue  # partial last line after a crash
        self._lock = threading.Lock()
        self.retried: dict[str, int] = {}  # reason -> retries (HTTP code or exception name)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
        )
        delay = 2.0
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code not in (408, 429, 500, 502, 503, 504) or attempt == self.retries - 1:
                    raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from None
                reason = str(e.code)
            # a body cut off mid-read (IncompleteRead, 2026-11-02) is a network error like the others
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                http.client.HTTPException,
                json.JSONDecodeError,
            ) as e:
                if attempt == self.retries - 1:
                    raise RuntimeError(repr(e)) from None
                reason = type(e).__name__
            with self._lock:
                self.retried[reason] = self.retried.get(reason, 0) + 1
            time.sleep(delay)
            delay *= 2
        raise RuntimeError("unreachable")

    def chat(self, messages: list[dict[str, str]], **params: Any) -> dict[str, Any]:
        key = request_key(self.model, messages, params)
        if key in self.cache:
            return self.cache[key]
        t0 = time.time()
        try:
            out = self._post({"model": self.model, "messages": messages, **params})
            ch = out["choices"][0]
            content = ch["message"].get("content") or ""
            rec = {
                "key": key,
                "model": out.get("model", self.model),
                "params": params,
                "content": THINK.sub("", content).strip(),
                "reasoning_chars": len(ch["message"].get("reasoning_content") or "")
                + (len(content) - len(THINK.sub("", content))),
                "finish_reason": ch.get("finish_reason"),
                "usage": out.get("usage"),
                "seconds": round(time.time() - t0, 2),
                "error": None,
                "task": self.task,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            }
        except RuntimeError as e:
            return {"key": key, "content": None, "error": str(e)}  # not cached: retried next run
        with self._lock:
            self.cache[key] = rec
            with self.cache_path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def map(
        self,
        requests: list[tuple[list[dict[str, str]], dict[str, Any]]],
        progress: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{}] * len(requests)

        def run(i: int) -> None:
            msgs, params = requests[i]
            out[i] = self.chat(msgs, **params)

        with ThreadPoolExecutor(self.concurrency) as ex:
            for n, _ in enumerate(ex.map(run, range(len(requests)))):
                if progress and (n + 1) % 1000 == 0:
                    progress(n + 1, len(requests))
        return out
