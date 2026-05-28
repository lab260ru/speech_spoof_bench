"""Badge-layer string builders (Phase 9).

Pure functions: input parsed dicts, output strings. No I/O.
"""

from __future__ import annotations

import json
from importlib import resources

import yaml
from jsonschema import ValidationError, validate


class BadgeError(Exception):
    """Raised on input that cannot produce a valid result.yaml or comment."""


def _color_for_eer(eer_percent: float) -> str:
    if eer_percent < 2.0:
        return "brightgreen"
    if eer_percent < 5.0:
        return "green"
    if eer_percent < 10.0:
        return "yellow"
    return "lightgrey"


def _load_result_schema() -> dict:
    with resources.files("speech_spoof_bench.schema").joinpath("result.schema.json").open("r") as f:
        return json.load(f)


def _project_submission_for_result(submission: dict, *, arena_url: str) -> dict:
    """Pure projection: submission dict → result.yaml dict.

    Raises BadgeError if required fields are missing.
    """
    try:
        system = submission["system"]
        dataset = submission["dataset"]
        scores = submission["scores"]
        artifact = submission["artifact"]
        slug = system["slug"]
    except KeyError as exc:
        raise BadgeError(f"submission missing required key: {exc.args[0]}") from exc

    return {
        "schema_version": 1,
        "system": {
            "name": system["name"],
            "slug": slug,
            "paper": {"arxiv_id": system["paper"]["arxiv_id"]},
        },
        "dataset": {
            "id": dataset["id"],
            "revision": dataset["revision"],
            "split": dataset["split"],
        },
        "scores": dict(scores),  # metric values + n_trials + n_skipped
        "arena": {
            "url": arena_url,
            "system_url": f"{arena_url}?system={slug}",
        },
        "artifact": {
            "scores_url": artifact["scores_url"],
        },
    }


def build_result_yaml(submission: dict, *, arena_url: str) -> str:
    """Render the paste-ready result.yaml string. Validates output against
    result.schema.json before returning.
    """
    projected = _project_submission_for_result(submission, arena_url=arena_url)
    try:
        validate(instance=projected, schema=_load_result_schema())
    except ValidationError as exc:
        raise BadgeError(f"result.yaml failed schema validation: {exc.message}") from exc
    return yaml.safe_dump(projected, sort_keys=False, default_flow_style=False)
