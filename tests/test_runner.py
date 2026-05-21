"""Tests for the runner (iterate + score + per-item fallback)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from speech_spoof_bench.loader import DatasetSource
from speech_spoof_bench.model import AntiSpoofingModel
from speech_spoof_bench.runner import TooManySkips, run_dataset


class _CountingModel(AntiSpoofingModel):
    """Returns sum of audio. Subclasses override _score_batch_hook."""

    name = "counting"
    batch_size = 4

    def __init__(self):
        self.load_count = 0
        self.unload_count = 0

    def load(self):
        self.load_count += 1

    def unload(self):
        self.unload_count += 1

    def score_batch(self, audios, srs):
        return [float(a.sum()) for a, s in zip(audios, srs)]


def _make_rows(n, sr=16000, bad_index=None):
    """Produce a synthetic IterableDataset-like list of rows."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        audio = rng.standard_normal(sr // 4).astype(np.float32) * 0.01
        row = {
            "path": f"a/{i}.wav",
            "audio": {"array": audio, "sampling_rate": sr},
            "label": i % 2,
            "notes": json.dumps({"utterance_id": f"UTT_{i:04d}"}),
        }
        if bad_index is not None and i == bad_index:
            row["_force_raise"] = True
        rows.append(row)
    return rows


def _source(slug="Synth"):
    return DatasetSource(
        spec=slug,
        display_name=slug,
        slug=slug,
        canonical_id=slug,
        metrics=["eer_percent"],
        split="test",
        is_local=True,
        local_path=None,
        revision=None,
    )


def test_run_dataset_writes_scores_and_labels(tmp_path):
    m = _CountingModel()
    m.load()
    res = run_dataset(m, _source(), _make_rows(8), tmp_path)
    m.unload()

    assert res.scores_path == tmp_path / "scores.txt"
    lines = res.scores_path.read_text().strip().splitlines()
    assert len(lines) == 8
    # Each line: "<utt_id> <score>"
    for line in lines:
        utt_id, score = line.split()
        assert utt_id.startswith("UTT_")
        float(score)  # parseable

    assert res.n_total == 8
    assert res.n_skipped == 0
    assert len(res.labels) == 8
    assert set(res.labels.values()) == {0, 1}


def test_per_item_skip_only_offender_in_multi_item_batch(tmp_path):
    class M(_CountingModel):
        def score_batch(self, audios, srs):
            # Raises if any item is "tagged" (contains NaN sentinel).
            if any(np.isnan(a).any() for a in audios):
                raise RuntimeError("batch contains bad item")
            return [float(a.sum()) for a in audios]

    rows = _make_rows(8)
    # Tag row 3 with a NaN-containing audio.
    rows[3]["audio"]["array"] = np.full(100, np.nan, dtype=np.float32)

    m = M()
    m.load()
    res = run_dataset(m, _source(), rows, tmp_path)
    m.unload()

    # Item 3 should be skipped; the other 7 scored successfully.
    assert res.n_total == 8
    assert res.n_skipped == 1
    lines = res.scores_path.read_text().strip().splitlines()
    assert len(lines) == 7
    assert "UTT_0003" not in res.scores_path.read_text()


def test_flaky_batch_recovers_via_single_item_calls(tmp_path):
    """Model fails on any batch > 1 but succeeds individually."""

    class M(_CountingModel):
        def score_batch(self, audios, srs):
            if len(audios) > 1:
                raise RuntimeError("only batch size 1 supported")
            return [float(audios[0].sum())]

    m = M()
    m.load()
    res = run_dataset(m, _source(), _make_rows(8), tmp_path)
    m.unload()

    assert res.n_total == 8
    assert res.n_skipped == 0
    lines = res.scores_path.read_text().strip().splitlines()
    assert len(lines) == 8


def test_too_many_skips_raises(tmp_path):
    class M(_CountingModel):
        def score_batch(self, audios, srs):
            raise RuntimeError("always broken")

    m = M()
    m.load()
    with pytest.raises(TooManySkips):
        run_dataset(m, _source(), _make_rows(20), tmp_path)
