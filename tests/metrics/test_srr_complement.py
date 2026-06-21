"""Tests for the srr_complement metric (1 - Spoof Rejection Rate, spoof-only-safe)."""

import numpy as np
import pytest

from speech_spoof_bench.metrics import get_metric, list_metrics
# Importing the metric module triggers its @register_metric decorator.
import speech_spoof_bench.metrics.srr_complement  # noqa: F401


def _spoof_inputs(spoof_scores):
    """All-spoof (single-class) inputs: every label is 1, no bonafide."""
    scores = {}
    labels = {}
    for i, s in enumerate(spoof_scores):
        utt_id = f"spoof_{i}"
        scores[utt_id] = float(s)
        labels[utt_id] = 1
    return scores, labels


def test_srr_complement_registered():
    spec = get_metric("srr_complement")
    assert spec.id == "srr_complement"
    assert spec.lower_is_better is True
    assert spec.requires_audio is False
    assert "srr_complement" in {m.id for m in list_metrics()}


def test_requires_threshold():
    scores, labels = _spoof_inputs([-1.0, 1.0])
    fn = get_metric("srr_complement").fn
    with pytest.raises(ValueError, match="threshold"):
        fn(scores, labels)  # no config
    with pytest.raises(ValueError, match="threshold"):
        fn(scores, labels, {})  # empty config


def test_all_spoof_below_threshold_is_zero():
    # Every spoof score < t* => SRR = 1 (all rejected) => value == 0.
    scores, labels = _spoof_inputs([-5.0] * 100)
    result = get_metric("srr_complement").fn(scores, labels, {"threshold": 0.0})
    assert result.value == pytest.approx(0.0, abs=1e-9)
    assert result.extras["n_spoof"] == 100


def test_all_spoof_above_threshold_is_hundred():
    # Every spoof score >= t* => SRR = 0 (none rejected) => value == 100.
    scores, labels = _spoof_inputs([5.0] * 100)
    result = get_metric("srr_complement").fn(scores, labels, {"threshold": 0.0})
    assert result.value == pytest.approx(100.0, abs=1e-9)


def test_spoof_only_does_not_raise_on_single_class():
    # The key spoof-only-safety property: all labels == 1 must NOT raise
    # (contrast eer_percent which raises with no bonafide class).
    scores, labels = _spoof_inputs([-1.0, 0.5, 2.0])
    result = get_metric("srr_complement").fn(scores, labels, {"threshold": 0.0})
    assert 0.0 <= result.value <= 100.0


def test_random_baseline_spoof_only_near_fifty():
    # Random N(0,1) scores at t*=0 => SRR ~ 0.5 => value ~ 50%
    # (the spoof-only analogue of the EER~50% smoke property).
    rng = np.random.default_rng(0)
    scores, labels = _spoof_inputs(rng.standard_normal(5000))
    result = get_metric("srr_complement").fn(scores, labels, {"threshold": 0.0})
    assert result.value == pytest.approx(50.0, abs=2.0)


def test_nan_spoof_score_raises():
    # A NaN score satisfies neither < t nor >= t -> it would silently bias the
    # rate. Reject non-finite spoof scores instead of producing a wrong number.
    scores, labels = _spoof_inputs([float("nan"), 1.0])
    with pytest.raises(ValueError):
        get_metric("srr_complement").fn(scores, labels, {"threshold": 0.0})


def test_inf_spoof_score_raises():
    scores, labels = _spoof_inputs([float("inf"), 1.0])
    with pytest.raises(ValueError):
        get_metric("srr_complement").fn(scores, labels, {"threshold": 0.0})


def test_nan_threshold_raises():
    # A NaN threshold makes (spoof < NaN) all-False -> value=100 silently.
    scores, labels = _spoof_inputs([-1.0, 1.0])
    with pytest.raises(ValueError):
        get_metric("srr_complement").fn(scores, labels, {"threshold": float("nan")})


def test_none_threshold_raises_valueerror():
    # None must raise a ValueError (bad config), not a raw TypeError from float().
    scores, labels = _spoof_inputs([-1.0, 1.0])
    with pytest.raises(ValueError):
        get_metric("srr_complement").fn(scores, labels, {"threshold": None})


def test_value_matches_manual():
    # spoof scores [-1, -1, +1, +1], t*=0 => two below (rejected), two above
    # => SRR = 0.5 => value == 50.0.
    scores, labels = _spoof_inputs([-1.0, -1.0, 1.0, 1.0])
    result = get_metric("srr_complement").fn(scores, labels, {"threshold": 0.0})
    assert result.value == pytest.approx(50.0, abs=1e-9)
    assert result.extras["srr"] == pytest.approx(0.5, abs=1e-9)
    assert result.extras["threshold"] == 0.0
