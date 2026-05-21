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


def fetch_manifest() -> dict[str, Any]:
    """Download manifest.yaml from HF, parse, validate, return dict.

    No auth required (public dataset repo).
    """
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
