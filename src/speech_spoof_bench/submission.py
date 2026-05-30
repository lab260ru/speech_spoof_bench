"""Submission YAML loader + schema validator.

Mirrors the shape of `manifest.py`. Public functions:
  - load_submission_schema()
  - parse_submission(text) -> dict
  - list_submission_files(dataset_id) -> list[str]
  - fetch_submission(dataset_id, path) -> dict
"""

from __future__ import annotations

import datetime as _dt
import json
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi
from jsonschema import ValidationError, validate

from . import hf_fetch

SCHEMA_PACKAGE = "speech_spoof_bench.schema"
SCHEMA_FILENAME = "submission.schema.json"
SUBMISSIONS_DIR = "submissions"
_EXCLUDED_FILENAMES = {"README.md", "results_template.yaml"}


class SubmissionValidationError(ValueError):
    """Raised when a submission YAML fails schema validation."""


def load_submission_schema() -> dict[str, Any]:
    with resources.files(SCHEMA_PACKAGE).joinpath(SCHEMA_FILENAME).open("r") as f:
        return json.load(f)


def _coerce_dates(obj: Any) -> Any:
    """Recursively turn datetime.date/datetime into ISO strings so jsonschema's
    `format: date` validator can accept PyYAML output."""
    if isinstance(obj, _dt.date):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _coerce_dates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_dates(v) for v in obj]
    return obj


def parse_submission(text: str) -> dict[str, Any]:
    """Parse YAML text and validate against the submission schema."""
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise SubmissionValidationError("submission YAML is not a mapping")
    data = _coerce_dates(data)
    try:
        validate(instance=data, schema=load_submission_schema())
    except ValidationError as exc:
        raise SubmissionValidationError(exc.message) from exc
    return data


def list_submission_files(dataset_id: str, *, api: HfApi | None = None) -> list[str]:
    """List `submissions/*.yaml` files in a dataset repo at main.

    Excludes README.md and results_template.yaml.
    """
    del api  # kept for compatibility with older tests/callers
    files = hf_fetch.list_repo_files(dataset_id, repo_type="dataset")
    out: list[str] = []
    for f in files:
        if not f.startswith(SUBMISSIONS_DIR + "/"):
            continue
        if not f.endswith(".yaml"):
            continue
        name = f.rsplit("/", 1)[-1]
        if name in _EXCLUDED_FILENAMES:
            continue
        out.append(f)
    return out


def fetch_submission(dataset_id: str, path: str) -> dict[str, Any]:
    """Download a submission YAML and parse+validate it."""
    local = hf_fetch.hub_download(repo_id=dataset_id, filename=path, repo_type="dataset")
    return parse_submission(Path(local).read_text())
