"""Persistence for nightly skip-unchanged.

A submission's reproduction result is a pure function of
``(scores_sha256, dataset.revision, installed bench_version)``. When all three
are unchanged since the last green verification, the result cannot have moved,
so nightly skips it. The store is a plain ``{submission_id: entry_key}`` JSON
dict persisted via ``actions/cache``; losing it only costs a safe re-verify.
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import __version__ as _BENCH_VERSION

DEFAULT_STORE_PATH = Path(".nightly-green.json")


def submission_id(dataset_id: str, slug: str) -> str:
    return f"{dataset_id}/{slug}"


def _entry_key(scores_sha256: str, revision: str) -> str:
    return f"{scores_sha256}|{revision}|{_BENCH_VERSION}"


def load(path: Path | str = DEFAULT_STORE_PATH) -> dict[str, str]:
    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text())


def save(store: dict[str, str], path: Path | str = DEFAULT_STORE_PATH) -> None:
    Path(path).write_text(json.dumps(store, indent=2, sort_keys=True))


def is_green(store: dict[str, str], dataset_id: str, slug: str,
             scores_sha256: str, revision: str) -> bool:
    return store.get(submission_id(dataset_id, slug)) == _entry_key(
        scores_sha256, revision
    )


def record_green(store: dict[str, str], dataset_id: str, slug: str,
                 scores_sha256: str, revision: str) -> None:
    store[submission_id(dataset_id, slug)] = _entry_key(scores_sha256, revision)
