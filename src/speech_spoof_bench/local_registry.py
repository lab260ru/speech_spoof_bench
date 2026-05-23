"""Local-dataset registry (Phase 8 §3).

Maps HF dataset ids to local on-disk parquet directories so commands that
would otherwise stream from HF can read from a local checkout instead.

Registry lives at the pip-package repo root: ``<repo>/local-datasets.yaml``.
Gitignored. Schema:

    schema_version: 1
    datasets:
      - {id: Org/Name, path: /abs/path}
      - ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

_REGISTRY_FILENAME = "local-datasets.yaml"
_SCHEMA_VERSION = 1


def _registry_path() -> Path:
    """Resolve the registry file location.

    For an editable install (``pip install -e .``), the package lives at
    ``<repo>/src/speech_spoof_bench``; the registry sits at ``<repo>/<file>``.
    For a non-editable install the path simply doesn't exist on disk, so
    ``load()`` returns an empty mapping and the loader proceeds as today.
    """
    return Path(__file__).resolve().parents[2] / _REGISTRY_FILENAME


def load() -> dict[str, str]:
    """Return ``{dataset_id: local_path_str}``. Empty dict if file missing."""
    p = _registry_path()
    if not p.is_file():
        return {}
    data = yaml.safe_load(p.read_text()) or {}
    entries = data.get("datasets", []) or []
    return {e["id"]: e["path"] for e in entries}


def _write(mapping: dict[str, str]) -> None:
    p = _registry_path()
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "datasets": [{"id": k, "path": v} for k, v in mapping.items()],
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(payload, sort_keys=False))


def _validate_dataset_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a directory")
    if not (path / "eval.yaml").is_file():
        raise ValueError(f"{path}/eval.yaml is missing")
    shards = list((path / "data").glob("test-*.parquet")) if (path / "data").is_dir() else []
    if not shards:
        raise ValueError(f"{path}/data/test-*.parquet not found")


def set(dataset_id: str, path: Path | str) -> None:
    """Register a local directory for the given dataset id. Validates first."""
    path = Path(path).expanduser().resolve()
    _validate_dataset_dir(path)
    mapping = load()
    mapping[dataset_id] = str(path)
    _write(mapping)


def unset(dataset_id: str) -> None:
    mapping = load()
    if dataset_id in mapping:
        del mapping[dataset_id]
        _write(mapping)


def lookup(dataset_id: str) -> Optional[Path]:
    """Return the registered local path, or None if unmapped.

    Raises FileNotFoundError if registered but path no longer exists — never
    silently falls back to HF.
    """
    mapping = load()
    if dataset_id not in mapping:
        return None
    p = Path(mapping[dataset_id])
    if not p.is_dir():
        raise FileNotFoundError(
            f"{dataset_id} is registered as local at {p}, but the directory "
            f"does not exist. Fix the path or unset the registration "
            f"(`speech-spoof-bench local unset {dataset_id}`)."
        )
    return p
