"""Validates result.schema.json's strictness boundary."""
from __future__ import annotations

import json
from importlib import resources

import pytest
from jsonschema import ValidationError, validate


def _schema():
    with resources.files("speech_spoof_bench.schema").joinpath("result.schema.json").open("r") as f:
        return json.load(f)


def _good():
    return {
        "schema_version": 1,
        "system": {
            "name": "AASIST",
            "slug": "aasist",
            "paper": {"arxiv_id": "2110.01200"},
        },
        "dataset": {
            "id": "Org/Foo",
            "revision": "abc1234",
            "split": "test",
        },
        "scores": {
            "n_trials": 1,
            "n_skipped": 0,
            "eer_percent": 1.23,
        },
        "arena": {
            "url": "https://huggingface.co/spaces/Org/Arena",
            "system_url": "https://huggingface.co/spaces/Org/Arena?system=aasist",
        },
        "artifact": {
            "scores_url": "https://huggingface.co/u/r/resolve/abc1234/.eval_results/Org/Foo/scores.txt",
        },
    }


def test_good_passes():
    validate(instance=_good(), schema=_schema())


def test_wrong_schema_version_rejected():
    d = _good(); d["schema_version"] = 2
    with pytest.raises(ValidationError):
        validate(instance=d, schema=_schema())


def test_extra_top_level_key_rejected():
    d = _good(); d["surprise"] = 1
    with pytest.raises(ValidationError):
        validate(instance=d, schema=_schema())


def test_missing_arena_system_url_rejected():
    d = _good(); del d["arena"]["system_url"]
    with pytest.raises(ValidationError):
        validate(instance=d, schema=_schema())


def test_non_https_arena_url_rejected():
    d = _good(); d["arena"]["url"] = "http://insecure.example"
    with pytest.raises(ValidationError):
        validate(instance=d, schema=_schema())


def test_scores_must_have_at_least_one_metric_besides_counts():
    d = _good(); d["scores"] = {"n_trials": 1, "n_skipped": 0}
    with pytest.raises(ValidationError):
        validate(instance=d, schema=_schema())
