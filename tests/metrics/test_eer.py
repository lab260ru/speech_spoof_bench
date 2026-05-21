"""Tests for the eer_percent metric."""

import numpy as np
import pytest

from speech_spoof_bench.metrics import get_metric, list_metrics
# Importing the metric module triggers its @register_metric decorator.
import speech_spoof_bench.metrics.eer  # noqa: F401


def _make_inputs(bonafide_scores, spoof_scores):
    scores = {}
    labels = {}
    for i, s in enumerate(bonafide_scores):
        utt_id = f"bona_{i}"
        scores[utt_id] = float(s)
        labels[utt_id] = 0
    for i, s in enumerate(spoof_scores):
        utt_id = f"spoof_{i}"
        scores[utt_id] = float(s)
        labels[utt_id] = 1
    return scores, labels


def test_eer_is_registered():
    spec = get_metric("eer_percent")
    assert spec.id == "eer_percent"
    assert spec.lower_is_better is True
    assert spec.requires_audio is False
    assert "eer_percent" in {m.id for m in list_metrics()}


def test_eer_perfectly_separable_is_zero():
    # Bonafide scores all higher than every spoof score → EER == 0.
    scores, labels = _make_inputs(
        bonafide_scores=np.linspace(10.0, 20.0, 500),
        spoof_scores=np.linspace(-20.0, -10.0, 500),
    )
    result = get_metric("eer_percent").fn(scores, labels)
    assert result.value == pytest.approx(0.0, abs=1e-9)
    assert result.extras["n_trials"] == 1000


def test_eer_fully_overlapping_is_near_fifty():
    # Same distribution → EER ≈ 50%.
    rng = np.random.default_rng(0)
    scores, labels = _make_inputs(
        bonafide_scores=rng.standard_normal(5000),
        spoof_scores=rng.standard_normal(5000),
    )
    result = get_metric("eer_percent").fn(scores, labels)
    assert result.value == pytest.approx(50.0, abs=2.0)


def test_eer_known_intermediate():
    # Shifted normals: theoretical EER for shift d=2 is ~15.87%
    # (Φ(-d/2) where Φ is the standard normal CDF). Allow some Monte Carlo slack.
    rng = np.random.default_rng(1)
    scores, labels = _make_inputs(
        bonafide_scores=rng.standard_normal(10000) + 1.0,
        spoof_scores=rng.standard_normal(10000) - 1.0,
    )
    result = get_metric("eer_percent").fn(scores, labels)
    assert result.value == pytest.approx(15.87, abs=1.5)
    assert "threshold" in result.extras


def test_eer_raises_when_no_bonafide():
    scores, labels = _make_inputs(bonafide_scores=[], spoof_scores=[1.0, 2.0])
    with pytest.raises(ValueError, match="bonafide"):
        get_metric("eer_percent").fn(scores, labels)


def test_eer_raises_when_no_spoof():
    scores, labels = _make_inputs(bonafide_scores=[1.0, 2.0], spoof_scores=[])
    with pytest.raises(ValueError, match="spoof"):
        get_metric("eer_percent").fn(scores, labels)
