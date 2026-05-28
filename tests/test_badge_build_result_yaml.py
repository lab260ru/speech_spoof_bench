"""build_result_yaml is a pure renderer; snapshot the YAML string."""
from __future__ import annotations

import pytest
import yaml

from speech_spoof_bench import badge


_ARENA_URL = "https://huggingface.co/spaces/Org/Arena"

_SUBMISSION = {
    "schema_version": 4,
    "system": {
        "name": "AASIST",
        "slug": "aasist",
        "description": "x",
        "code": "https://x",
        "checkpoint": "https://x",
        "paper": {
            "arxiv_id": "2110.01200",
            "url": "https://arxiv.org/abs/2110.01200",
            "bibtex": "@x{1, }",
        },
    },
    "dataset": {"id": "Org/Foo", "revision": "abc1234", "split": "test"},
    "scores": {"eer_percent": 1.23, "n_trials": 71237, "n_skipped": 0},
    "artifact": {
        "scores_url": "https://huggingface.co/u/r/resolve/abc1234/.eval_results/Org/Foo/scores.txt",
        "scores_sha256": "0" * 64,
        "bench_version": "speech-spoof-bench==0.1.0",
    },
    "submitter": {"hf_username": "u", "contact": "u@example.com"},
    "submitted_at": "2026-05-23",
}


def test_returns_string_parseable_as_mapping():
    out = badge.build_result_yaml(_SUBMISSION, arena_url=_ARENA_URL)
    assert isinstance(out, str)
    parsed = yaml.safe_load(out)
    assert isinstance(parsed, dict)


def test_required_top_level_keys_present():
    parsed = yaml.safe_load(badge.build_result_yaml(_SUBMISSION, arena_url=_ARENA_URL))
    assert set(parsed.keys()) == {
        "schema_version", "system", "dataset", "scores", "arena", "artifact",
    }


def test_drops_submission_only_fields():
    parsed = yaml.safe_load(badge.build_result_yaml(_SUBMISSION, arena_url=_ARENA_URL))
    assert "submitter" not in parsed
    assert "reproduction" not in parsed
    assert "submitted_at" not in parsed
    # artifact loses scores_sha256 and bench_version too
    assert set(parsed["artifact"].keys()) == {"scores_url"}
    # system keeps name/slug/paper.arxiv_id only
    assert set(parsed["system"].keys()) == {"name", "slug", "paper"}
    assert set(parsed["system"]["paper"].keys()) == {"arxiv_id"}


def test_scores_preserves_counts_and_metrics():
    parsed = yaml.safe_load(badge.build_result_yaml(_SUBMISSION, arena_url=_ARENA_URL))
    assert parsed["scores"] == {"eer_percent": 1.23, "n_trials": 71237, "n_skipped": 0}


def test_arena_urls_constructed_from_slug():
    parsed = yaml.safe_load(badge.build_result_yaml(_SUBMISSION, arena_url=_ARENA_URL))
    assert parsed["arena"]["url"] == _ARENA_URL
    assert parsed["arena"]["system_url"] == f"{_ARENA_URL}?system=aasist"


def test_schema_version_is_one_not_four():
    parsed = yaml.safe_load(badge.build_result_yaml(_SUBMISSION, arena_url=_ARENA_URL))
    assert parsed["schema_version"] == 1  # result.yaml is its own schema, starting at 1


def test_invalid_submission_raises_badge_error():
    bad = {"schema_version": 4}  # missing required fields
    with pytest.raises(badge.BadgeError):
        badge.build_result_yaml(bad, arena_url=_ARENA_URL)


def test_output_passes_result_schema():
    """The whole point: rendered output must validate against result.schema.json."""
    import json
    from importlib import resources
    from jsonschema import validate

    with resources.files("speech_spoof_bench.schema").joinpath("result.schema.json").open("r") as f:
        schema = json.load(f)

    out = badge.build_result_yaml(_SUBMISSION, arena_url=_ARENA_URL)
    parsed = yaml.safe_load(out)
    validate(instance=parsed, schema=schema)
