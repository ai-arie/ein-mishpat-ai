"""Write an eval run's resolved config + results next to its outputs and log it to MLflow."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

import yaml

from ein_mishpat_ai.utils.code_hash import code_hash
from ein_mishpat_ai.utils.io import now_iso, project_root


def environment() -> dict[str, str]:
    import sentence_transformers
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def write_run(
    out_dir: Path,
    experiment: str,
    run_name: str,
    config: dict[str, Any],
    metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "experiment": experiment,
        "run_name": run_name,
        "finished_at": now_iso(),
        "code_hash": code_hash(),
        "environment": environment(),
        "config": config,
        "metrics": metrics,
        **(extra or {}),
    }
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    )
    (out_dir / "results.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))

    import mlflow

    if mlflow.get_experiment_by_name(experiment) is None:
        try:
            mlflow.create_experiment(
                experiment, artifact_location=(project_root() / "mlruns" / "artifacts").as_uri()
            )
        except mlflow.exceptions.MlflowException:  # created concurrently by another job
            pass
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("code_hash", record["code_hash"])
        flat = {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool))}
        mlflow.log_params(flat)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.log_artifact(str(out_dir / "results.json"))
    return record
