"""arena-manifest reader.

Fetches the single-file manifest from the public HF dataset repo
`SpeechAntiSpoofingBenchmarks/arena-manifest`, validates it against the
bundled JSON Schema, and exposes a few small accessors.
"""

from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from jsonschema import validate

from .hf_fetch import hub_download as hf_hub_download

MANIFEST_REPO = "SpeechAntiSpoofingBenchmarks/arena-manifest"
MANIFEST_FILENAME = "manifest.yaml"
MANIFEST_REPO_ENV = "SSB_ARENA_MANIFEST_REPO"
MANIFEST_REVISION_ENV = "SSB_ARENA_MANIFEST_REVISION"
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


def fetch_manifest(
    repo_id: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Download manifest.yaml from HF, parse, validate, return dict.

    No auth required for the public production repo. Staging Spaces can point
    at another manifest repo/ref via SSB_ARENA_MANIFEST_REPO and
    SSB_ARENA_MANIFEST_REVISION.
    """
    repo_id = repo_id or os.environ.get(MANIFEST_REPO_ENV) or MANIFEST_REPO
    revision = revision if revision is not None else os.environ.get(MANIFEST_REVISION_ENV)
    local = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=MANIFEST_FILENAME,
        revision=revision,
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
