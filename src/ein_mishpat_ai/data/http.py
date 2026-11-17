"""HTTP helpers: a shared session per thread, retries with backoff, and a cross-process API rate limit."""

from __future__ import annotations

import fcntl
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from ein_mishpat_ai.utils.io import project_root

RETRY_STATUS = {429, 500, 502, 503, 504}

_local = threading.local()


def session(user_agent: str) -> requests.Session:
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = user_agent
        adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
        s.mount("https://", adapter)
        _local.session = s
    return s


class FileRateLimiter:
    """At most `rate` requests per second across all processes that share `lock_path`.

    The timestamp of the last request lives in the lock file itself, guarded by flock.
    """

    def __init__(self, rate: float, lock_path: Path | None = None):
        self.interval = 1.0 / rate
        self.lock_path = lock_path or project_root() / ".tmp" / "sefaria_api.ratelimit"
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.touch(exist_ok=True)

    def wait(self) -> None:
        with self.lock_path.open("r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                raw = f.read().strip()
                last = float(raw) if raw else 0.0
                delay = last + self.interval - time.time()
                if delay > 0:
                    time.sleep(delay)
                f.seek(0)
                f.truncate()
                f.write(f"{time.time():.6f}")
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


@dataclass
class FetchResult:
    status: int  # HTTP status, or 0 for a network error
    content: bytes | None
    error: str | None = None


def fetch(
    url: str,
    *,
    user_agent: str,
    retries: int = 5,
    timeout: float = 120,
    limiter: FileRateLimiter | None = None,
    params: dict | None = None,
) -> FetchResult:
    """GET into memory. Retries 429/5xx and network errors with exponential backoff + jitter."""
    last = FetchResult(0, None, "not attempted")
    for attempt in range(retries):
        if limiter:
            limiter.wait()
        try:
            r = session(user_agent).get(url, timeout=timeout, params=params)
        except requests.RequestException as e:
            last = FetchResult(0, None, f"{type(e).__name__}: {e}"[:300])
        else:
            if r.status_code == 200:
                return FetchResult(200, r.content)
            last = FetchResult(r.status_code, None, f"http_{r.status_code}")
            if r.status_code not in RETRY_STATUS:
                return last
            retry_after = r.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                time.sleep(min(float(retry_after), 120))
                continue
        if attempt < retries - 1:
            time.sleep(min(2**attempt, 60) + random.uniform(0, 1))
    return last
