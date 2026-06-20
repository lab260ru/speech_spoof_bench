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
        "ranking_version": "v1", "schema_version": 2,
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


def test_tier_with_requires_paper_valid():
    validate(_m(tiers=[
        {"name": "gold", "min_coverage": 1.0, "requires_paper": True},
        {"name": "unpublished", "min_coverage": 0.0, "requires_paper": False},
    ]), _schema())


def test_non_boolean_requires_paper_rejected():
    with pytest.raises(ValidationError):
        validate(_m(tiers=[{"name": "gold", "min_coverage": 1.0, "requires_paper": "yes"}]), _schema())


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


# --- manifest schema 1 -> 2: dataset_entry fields + top-level overrides ----

def _entry(**extra):
    return {"id": "Org/A", "revision": "abc1234", **extra}


def test_dataset_entry_accepts_status_and_saturated_at():
    validate(_m({"core_set": [_entry(status="saturated", saturated_at="2026-06-20")]}), _schema())


def test_dataset_entry_accepts_status_active():
    validate(_m({"core_set": [_entry(status="active")]}), _schema())


def test_dataset_entry_rejects_bad_status():
    with pytest.raises(ValidationError):
        validate(_m({"core_set": [_entry(status="frozen")]}), _schema())


def test_dataset_entry_accepts_penalty_number():
    validate(_m({"core_set": [_entry(penalty=30.0)]}), _schema())


def test_dataset_entry_rejects_non_number_penalty():
    with pytest.raises(ValidationError):
        validate(_m({"core_set": [_entry(penalty="high")]}), _schema())


def test_dataset_entry_accepts_metric_and_category():
    validate(_m({"core_set": [_entry(metric="srr_complement", category="spoof_only")]}), _schema())


def test_dataset_entry_rejects_bad_category():
    with pytest.raises(ValidationError):
        validate(_m({"core_set": [_entry(category="weird")]}), _schema())


def test_dataset_entry_rejects_empty_metric():
    with pytest.raises(ValidationError):
        validate(_m({"core_set": [_entry(metric="")]}), _schema())


def test_dataset_entry_accepts_verification_enum():
    for v in ("reproduce", "organizer", "scores-only"):
        validate(_m({"core_set": [_entry(verification=v)]}), _schema())


def test_dataset_entry_rejects_bad_verification():
    with pytest.raises(ValidationError):
        validate(_m({"core_set": [_entry(verification="nope")]}), _schema())


def test_accepts_contamination_overrides():
    validate(_m({"contamination_overrides": {"some-slug": ["Org/A", "Org/B"]}}), _schema())


def test_rejects_non_list_contamination_overrides_value():
    with pytest.raises(ValidationError):
        validate(_m({"contamination_overrides": {"some-slug": "not-a-list"}}), _schema())


def test_platinum_tier_valid():
    validate(_m(tiers=[
        {"name": "platinum",    "min_coverage": 1.0,  "color": "#E5E4E2", "requires_paper": True},
        {"name": "gold",        "min_coverage": 0.85, "color": "#FFD700", "requires_paper": True},
        {"name": "silver",      "min_coverage": 0.6,  "color": "#C0C0C0", "requires_paper": True},
        {"name": "bronze",      "min_coverage": 0.0,  "color": "#CD7F32", "requires_paper": True},
        {"name": "unpublished", "min_coverage": 0.0,  "color": "#9aa0a6", "requires_paper": False},
    ]), _schema())


def test_rejects_schema_version_1():
    # const bumped 1 -> 2: old manifests are now rejected.
    with pytest.raises(ValidationError):
        validate(_m({"schema_version": 1}), _schema())


def test_const_is_now_2():
    assert _schema()["properties"]["schema_version"]["const"] == 2
