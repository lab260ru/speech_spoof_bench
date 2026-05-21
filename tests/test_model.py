"""Tests for the AntiSpoofingModel ABC and the Simple helper."""

import numpy as np
import pytest

from speech_spoof_bench.model import AntiSpoofingModel, SimpleAntiSpoofingModel


def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        AntiSpoofingModel()  # type: ignore[abstract]


def test_cannot_instantiate_simple_without_score():
    with pytest.raises(TypeError):
        SimpleAntiSpoofingModel()  # type: ignore[abstract]


def test_simple_model_score_batch_falls_back_to_score():
    class M(SimpleAntiSpoofingModel):
        name = "test"
        def load(self):
            pass
        def score(self, audio, sr):
            return float(audio.sum())

    m = M()
    m.load()
    out = m.score_batch(
        [np.array([1.0, 2.0]), np.array([3.0])],
        [16000, 16000],
    )
    assert out == [3.0, 3.0]


def test_model_defaults():
    class M(SimpleAntiSpoofingModel):
        name = "test"
        def load(self):
            pass
        def score(self, audio, sr):
            return 0.0

    m = M()
    assert m.expected_sample_rate == 16000
    assert m.batch_size == 1
