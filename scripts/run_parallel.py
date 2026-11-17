"""Run a list of jobs, each on its own GPU(s) via CUDA_VISIBLE_DEVICES.

usage: python scripts/run_parallel.py <jobs.yaml> [--gpus 0,1,2,3,4,5,6,7] [--force]

jobs.yaml:
  batch: m0_sefaria_bench          # logs/<batch>/<job>.log, logs/<batch>/status.tsv
  jobs:
    - {name: f2llm-1.7b, cmd: "python scripts/eval_sefaria_bench.py f2llm-1.7b", gpus: 1}

Resumable: a job whose last status is `done` is skipped unless --force. Start the runner itself
detached (for example with nohup or tmux).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def last_status(status_file: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if status_file.exists():
        for line in status_file.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                out[parts[1]] = parts[2]
    return out


def log_status(status_file: Path, name: str, status: str, detail: str = "") -> None:
    with status_file.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{name}\t{status}\t{detail}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.jobs).read_text())
    log_dir = ROOT / "logs" / spec["batch"]
    log_dir.mkdir(parents=True, exist_ok=True)
    status_file = log_dir / "status.tsv"
    done = {n for n, s in last_status(status_file).items() if s == "done"}
    queue = [j for j in spec["jobs"] if args.force or j["name"] not in done]
    free = [g.strip() for g in args.gpus.split(",") if g.strip()]
    running: dict[str, tuple[subprocess.Popen, list[str], float]] = {}
    print(f"{len(queue)} jobs to run ({len(spec['jobs']) - len(queue)} already done), gpus={free}")

    while queue or running:
        for name, (proc, gpus, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            status = "done" if rc == 0 else "failed"
            log_status(status_file, name, status, f"rc={rc} minutes={(time.time() - t0) / 60:.1f}")
            print(f"{name}: {status} (rc={rc})", flush=True)
            free.extend(gpus)
            del running[name]
        while queue and len(free) >= queue[0].get("gpus", 1):
            job = queue.pop(0)
            n = job.get("gpus", 1)
            gpus, free = free[:n], free[n:]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
            log = (log_dir / f"{job['name']}.log").open("w")
            proc = subprocess.Popen(
                job["cmd"], shell=True, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            running[job["name"]] = (proc, gpus, time.time())
            log_status(status_file, job["name"], "started", f"gpus={','.join(gpus)} pid={proc.pid}")
            print(f"{job['name']}: started on gpus {gpus}", flush=True)
        if queue and not running and len(free) < queue[0].get("gpus", 1):
            raise SystemExit(f"job {queue[0]['name']} needs more GPUs than available")
        time.sleep(5)


if __name__ == "__main__":
    main()
