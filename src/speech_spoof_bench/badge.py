"""Badge-layer string builders (Phase 9).

Pure functions: input parsed dicts, output strings. No I/O.
"""

from __future__ import annotations

import json
from importlib import resources
from urllib.parse import quote

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


def _shields_url(metric: str, dataset_name: str, value: float, color: str) -> str:
    """Build a static shields.io badge URL.

    Shields rules: `_` → real underscore, `__` → literal `_`. So pre-double
    underscores in our segment text, then URL-encode the result.
    """
    label = quote(f"{metric.upper().replace('_', ' ')} on {dataset_name}".replace("_", "__"))
    msg = quote(f"{value}%".replace("_", "__"))
    return f"https://img.shields.io/badge/{label}-{msg}-{color}"


def _metric_display_for(metric: str) -> str:
    """`eer_percent` → `EER%` (the value already carries the % suffix)."""
    if metric == "eer_percent":
        return "EER%"
    return metric.upper().replace("_", " ")


def build_paste_comment(
    submission: dict,
    *,
    arena_url: str,
    dataset_canonical_id: str,
    primary_metric: str,
    submission_path: str,
    merge_sha: str,
    gh_run_url: str,
) -> str:
    """Render the full Markdown comment body for the post-merge badge step."""
    scores = submission.get("scores", {})
    if primary_metric not in scores:
        raise BadgeError(
            f"primary metric {primary_metric!r} missing from submission scores"
        )
    value = scores[primary_metric]
    slug = submission["system"]["slug"]
    dataset_name = dataset_canonical_id.split("/", 1)[1]
    color = _color_for_eer(value) if primary_metric == "eer_percent" else "blue"

    result_yaml = build_result_yaml(submission, arena_url=arena_url)
    metric_label = _metric_display_for(primary_metric)
    shields_url = _shields_url(metric_label, dataset_name, value, color)
    badge_md = (
        f"[![{metric_label} {value} on {dataset_name}]"
        f"({shields_url})]"
        f"({arena_url}?system={slug})"
    )

    return (
        f"**speech-spoof-bench** — submission merged ✅\n\n"
        f"System `{slug}` is now live on the [Arena]({arena_url}?system={slug}).\n\n"
        f"To display a backlink badge on your model page, take the two steps below.\n\n"
        f"### 1. Add `result.yaml` to your model repo\n\n"
        f"```yaml\n{result_yaml}```\n\n"
        f"Upload it with:\n\n"
        f"```bash\n"
        f"huggingface-cli upload <your-model-repo> result.yaml \\\n"
        f"  .eval_results/{dataset_canonical_id}/result.yaml\n"
        f"```\n\n"
        f"### 2. Add the badge line to your README\n\n"
        f"```markdown\n{badge_md}\n```\n\n"
        f"<!-- ssb:badge --> sha={merge_sha} path={submission_path}\n\n"
        f"---\n"
        f"_🤖 [view CI run]({gh_run_url})_\n"
    )
