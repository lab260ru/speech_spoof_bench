"""manifest.schema.json accepts the optional ranking block + tier color."""
from __future__ import annotations

import json
from importlib import resources

import pytest
from jsonschema import ValidationError, validate


def _schema():
    with resources.files("speech_spoof_bench.schema").joinpath("manifest.schema.json").open("r") as f:
        return json.load(f)


def _m(extra=None, tiers=None):
    base = {
        "ranking_version": "v1", "schema_version": 1,
        "metrics_in_use": ["eer_percent"],
        "tiers": tiers or [{"name": "gold", "min_coverage": 1.0}],
        "core_set": [{"id": "Org/A", "revision": "abc1234"}],
        "extended": [],
    }
    if extra:
        base.update(extra)
    return base


def test_manifest_without_ranking_block_valid():
    validate(_m(), _schema())


def test_manifest_with_full_ranking_block_valid():
    validate(_m({"ranking": {
        "metric": "eer_percent", "absence_penalty": 50.0,
        "gamma_aggregated": 0.0, "gamma_pooled": 1.0,
        "default_view": "aggregated", "weights": {"Org/A": 1.0},
    }}), _schema())


def test_tier_with_color_valid():
    validate(_m(tiers=[{"name": "gold", "min_coverage": 1.0, "color": "#FFD700"}]), _schema())


def test_bad_default_view_rejected():
    with pytest.raises(ValidationError):
        validate(_m({"ranking": {"default_view": "sideways"}}), _schema())


def test_negative_gamma_rejected():
    with pytest.raises(ValidationError):
        validate(_m({"ranking": {"gamma_pooled": -1.0}}), _schema())


def test_non_number_penalty_rejected():
    with pytest.raises(ValidationError):
        validate(_m({"ranking": {"absence_penalty": "high"}}), _schema())


def test_unknown_ranking_field_rejected():
    with pytest.raises(ValidationError):
        validate(_m({"ranking": {"bogus": 1}}), _schema())
