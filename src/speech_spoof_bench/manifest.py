"""arena-manifest reader.

Fetches the single-file manifest from the public HF dataset repo
`SpeechAntiSpoofingBenchmarks/arena-manifest`, validates it against the
bundled JSON Schema, and exposes a few small accessors.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import hf_hub_download
from jsonschema import validate

from .ci._hf_retry import retry_on_429

MANIFEST_REPO = "SpeechAntiSpoofingBenchmarks/arena-manifest"
MANIFEST_FILENAME = "manifest.yaml"
SCHEMA_PACKAGE = "speech_spoof_bench.schema"
SCHEMA_FILENAME = "manifest.schema.json"


def _load_schema() -> dict[str, Any]:
    with resources.files(SCHEMA_PACKAGE).joinpath(SCHEMA_FILENAME).open("r") as f:
        return json.load(f)


def _parse_and_validate(text: str) -> dict[str, Any]:
    data = yaml.safe_load(text)
    validate(instance=data, schema=_load_schema())
    return data


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load + validate a local manifest file. Used in tests and offline dev."""
    return _parse_and_validate(Path(path).read_text())


def fetch_manifest(*, retry: bool = True) -> dict[str, Any]:
    """Download manifest.yaml from HF, parse, validate, return dict.

    No auth required (public dataset repo). With ``retry=True`` (default) the
    download is wrapped in ``retry_on_429`` so a transient HF throttle self-heals
    instead of crashing every consumer — most importantly the nightly
    revalidation and the `ci sweep` backstop, both background jobs that can
    afford to wait.

    ``retry=False`` does a single shot with no backoff: for **request-path**
    callers that must not block (the Arena webhook's subscription gate runs
    synchronously inside the async handler — a multi-minute retry budget there
    re-creates HF's 3-strikes webhook auto-disable).
    """
    if retry:
        local = retry_on_429(
            hf_hub_download,
            repo_id=MANIFEST_REPO,
            repo_type="dataset",
            filename=MANIFEST_FILENAME,
        )
    else:
        local = hf_hub_download(
            repo_id=MANIFEST_REPO,
            repo_type="dataset",
            filename=MANIFEST_FILENAME,
        )
    return _parse_and_validate(Path(local).read_text())


def core_dataset_ids(manifest: dict[str, Any]) -> list[str]:
    return [entry["id"] for entry in manifest["core_set"]]


def all_dataset_ids(manifest: dict[str, Any]) -> list[str]:
    return [entry["id"] for entry in manifest["core_set"] + manifest["extended"]]


def revision_for(manifest: dict[str, Any], dataset_id: str) -> str | None:
    for entry in manifest["core_set"] + manifest["extended"]:
        if entry["id"] == dataset_id:
            return entry["revision"]
    return None
