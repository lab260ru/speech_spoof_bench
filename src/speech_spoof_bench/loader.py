"""Dataset loader — local directory or HF repo id."""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from datasets import Audio, Features, Value, load_dataset
from huggingface_hub import hf_hub_download

from .metrics import is_registered

_HF_ID_RE = re.compile(r"^[^/]+/[^/]+$")


@dataclass(frozen=True)
class DatasetSource:
    spec: str
    display_name: str
    slug: str
    canonical_id: str
    metrics: list[str]
    split: str
    is_local: bool
    local_path: Path | None
    revision: str | None


def _parse_eval_yaml(eval_path: Path) -> dict[str, Any]:
    if not eval_path.is_file():
        raise FileNotFoundError(f"missing eval.yaml at {eval_path}")
    data = yaml.safe_load(eval_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"eval.yaml at {eval_path} is not a mapping")
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"eval.yaml at {eval_path} missing non-empty 'name'")
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"eval.yaml at {eval_path} missing 'tasks' list")
    task = tasks[0]
    split = task.get("split", "test")
    metrics = task.get("metrics", [])
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(f"eval.yaml at {eval_path} missing task[0].metrics")
    for m in metrics:
        if not is_registered(m):
            raise KeyError(f"metric id {m!r} not registered (from {eval_path})")
    return {"name": name, "split": split, "metrics": metrics}


def _resolve_local(path: Path, streaming: bool):
    meta = _parse_eval_yaml(path / "eval.yaml")
    shards = sorted(glob.glob(str(path / "data" / "test-*.parquet")))
    if not shards:
        raise FileNotFoundError(
            f"no parquet shards under {path / 'data'} matching test-*.parquet"
        )

    # Force the audio column to decode as Audio so .array / .sampling_rate work.
    features = Features(
        {
            "path": Value("string"),
            "audio": Audio(sampling_rate=None),
            "label": Value("int64"),
            "notes": Value("string"),
        }
    )
    ds = load_dataset(
        "parquet",
        data_files={"train": shards},
        split="train",
        streaming=streaming,
        features=features,
    )

    source = DatasetSource(
        spec=str(path),
        display_name=meta["name"],
        slug=path.name,
        canonical_id=path.name,
        metrics=list(meta["metrics"]),
        split=meta["split"],
        is_local=True,
        local_path=path,
        revision=None,
    )
    return source, ds


def _resolve_hf(repo_id: str, streaming: bool):
    eval_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="eval.yaml",
            repo_type="dataset",
        )
    )
    meta = _parse_eval_yaml(eval_path)
    ds = load_dataset(repo_id, split=meta["split"], streaming=streaming)
    source = DatasetSource(
        spec=repo_id,
        display_name=meta["name"],
        slug=repo_id.split("/")[-1],
        canonical_id=repo_id,
        metrics=list(meta["metrics"]),
        split=meta["split"],
        is_local=False,
        local_path=None,
        # TODO(phase-4): resolve revision via arena-manifest.
        revision=None,
    )
    return source, ds


def resolve(spec: str, *, streaming: bool = True):
    """Resolve a dataset spec to a (DatasetSource, IterableDataset).

    Dispatch:
      1. If ``Path(spec).is_dir()`` → local mode.
      2. Else if ``spec`` looks like ``org/name`` → HF mode.
      3. Else → ValueError.
    """
    candidate_path = Path(spec)
    if candidate_path.is_dir():
        return _resolve_local(candidate_path, streaming)
    if _HF_ID_RE.match(spec):
        return _resolve_hf(spec, streaming)
    raise ValueError(
        f"dataset spec {spec!r} is not a directory and not in <org>/<name> form"
    )
