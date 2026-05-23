"""Phase 7b — `submit` command implementation.

Public surface (used by cli.py):
  - load_meta(path) -> dict
  - submit(...)             # added later in Task 8
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi
from jsonschema import ValidationError, validate

_META_SCHEMA_PACKAGE = "speech_spoof_bench.data"
_META_SCHEMA_FILENAME = "submission_meta.schema.json"


class MetaValidationError(ValueError):
    """Raised when a submission meta YAML fails schema validation."""


def _load_meta_schema() -> dict[str, Any]:
    with resources.files(_META_SCHEMA_PACKAGE).joinpath(_META_SCHEMA_FILENAME).open("r") as f:
        return json.load(f)


def load_meta(path: Path | str) -> dict[str, Any]:
    """Parse and validate a submission meta YAML.

    Raises:
      FileNotFoundError: path doesn't exist.
      MetaValidationError: YAML parses but fails the schema.
    """
    p = Path(path)
    text = p.read_text()  # raises FileNotFoundError as desired
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise MetaValidationError(f"{p}: not a YAML mapping")
    try:
        validate(instance=data, schema=_load_meta_schema())
    except ValidationError as exc:
        raise MetaValidationError(f"{p}: {exc.message}") from exc
    return data


def build_submission_payload(
    *,
    result_yaml: dict[str, Any],
    meta: dict[str, Any],
    scores_url: str,
    scores_sha256: str,
    hf_username: str,
    contact: str,
    submitted_at: str,
) -> dict[str, Any]:
    """Merge a result.yaml + meta into a fully-formed v4 submission dict.

    The `reproduction` block is left empty by design (§1.7) — the maintainer
    fills it via `reproduce --scoring` at merge time.
    """
    sys_meta = meta["system"]
    payload: dict[str, Any] = {
        "schema_version": 4,
        "system": {
            "name": sys_meta["name"],
            "slug": sys_meta["slug"],
            "description": sys_meta["description"],
            "code": sys_meta["code"],
            "checkpoint": sys_meta["checkpoint"],
            "paper": dict(sys_meta["paper"]),
        },
        "dataset": dict(result_yaml["dataset"]),
        "scores": dict(result_yaml["scores"]),
        "artifact": {
            "scores_url": scores_url,
            "scores_sha256": scores_sha256,
            "bench_version": result_yaml["artifact"]["bench_version"],
        },
        "reproduction": {},
        "submitter": {"hf_username": hf_username, "contact": contact},
        "submitted_at": submitted_at,
    }
    if "notes" in meta:
        payload["notes"] = meta["notes"]
    return payload


def upload_scores(
    *,
    api: HfApi,
    model_repo: str,
    dataset_canonical_id: str,
    local_path: Path | str,
) -> tuple[str, str]:
    """Upload scores.txt to the model repo's main branch.

    Returns (scores_url, commit_oid). The URL pins the returned commit oid.
    """
    p = Path(local_path)
    if not p.is_file():
        raise FileNotFoundError(f"scores file not found: {p}")
    path_in_repo = f".eval_results/{dataset_canonical_id}/scores.txt"
    info = api.upload_file(
        path_or_fileobj=str(p),
        path_in_repo=path_in_repo,
        repo_id=model_repo,
        repo_type="model",
        commit_message=f"upload scores for {dataset_canonical_id}",
    )
    oid = info.oid
    url = f"https://huggingface.co/{model_repo}/resolve/{oid}/{path_in_repo}"
    return url, oid
