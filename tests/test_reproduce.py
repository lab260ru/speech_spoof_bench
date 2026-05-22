"""Tests for reproduce.py (--scoring path)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import yaml

from speech_spoof_bench import reproduce

FIX = Path(__file__).parent / "fixtures"


def _patch_yaml(tmp_path, scores_path, scores_sha):
    src = (FIX / "submissions" / "valid.yaml").read_text()
    data = yaml.safe_load(src)
    data["artifact"]["scores_sha256"] = scores_sha
    data["scores"] = {"eer_percent": 25.0, "n_trials": 4, "n_skipped": 0}
    p = tmp_path / "submission.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_sha_mismatch_fails(tmp_path):
    fake = tmp_path / "scores.txt"
    fake.write_text("UTT_0000 1.0\n")
    real_sha = hashlib.sha256(fake.read_bytes()).hexdigest()
    # Claimed sha differs from observed; observed is real_sha but claimed is all-a.
    claimed_sha = "a" * 64
    yaml_path = _patch_yaml(tmp_path, fake, claimed_sha)
    with patch("speech_spoof_bench.reproduce.hf_fetch.download",
               return_value=(fake, real_sha)):
        rc = reproduce.run_scoring(yaml_path, tolerance=1e-6)
    assert rc != 0


def test_scores_parse(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("a 1.0\nb -0.5\n\n")
    parsed = reproduce._parse_scores_txt(p)
    assert parsed == {"a": 1.0, "b": -0.5}


def test_select_columns_called_before_iteration():
    """Guards the audio-not-downloaded invariant."""
    captured: dict = {}

    class FakeDS:
        def __init__(self, rows):
            self._rows = rows
            self.select_called_with = None
        def select_columns(self, cols):
            self.select_called_with = list(cols)
            captured["select"] = list(cols)
            return self
        def __iter__(self):
            captured["iter_after_select"] = "select" in captured
            return iter(self._rows)

    fake = FakeDS([
        {"notes": '{"utterance_id":"a"}', "label": 0},
        {"notes": '{"utterance_id":"b"}', "label": 1},
    ])
    with patch("speech_spoof_bench.reproduce.load_dataset", return_value=fake):
        labels = reproduce._stream_labels("x/y", "test", "deadbeef")
    assert captured["select"] == ["notes", "label"]
    assert captured["iter_after_select"] is True
    assert labels == {"a": 0, "b": 1}


def test_coverage_missing_in_dataset(tmp_path):
    scores = tmp_path / "s.txt"
    scores.write_text("UTT_0000 1.0\nGHOST 0.0\n")
    sha = hashlib.sha256(scores.read_bytes()).hexdigest()
    yaml_path = _patch_yaml(tmp_path, scores, sha)
    fake_labels = {"UTT_0000": 0, "UTT_0001": 1}
    with patch("speech_spoof_bench.reproduce.hf_fetch.download",
               return_value=(scores, sha)):
        rc = reproduce.run_scoring(
            yaml_path, label_stream=lambda *a, **k: fake_labels
        )
    assert rc == 1


def test_n_trials_mismatch(tmp_path):
    scores = tmp_path / "s.txt"
    scores.write_text("UTT_0000 1.0\n")
    sha = hashlib.sha256(scores.read_bytes()).hexdigest()
    src = (FIX / "submissions" / "valid.yaml").read_text()
    data = yaml.safe_load(src)
    data["artifact"]["scores_sha256"] = sha
    data["scores"] = {"eer_percent": 25.0, "n_trials": 999, "n_skipped": 0}
    p = tmp_path / "submission.yaml"
    p.write_text(yaml.safe_dump(data))
    fake_labels = {"UTT_0000": 0}
    with patch("speech_spoof_bench.reproduce.hf_fetch.download",
               return_value=(scores, sha)):
        rc = reproduce.run_scoring(p, label_stream=lambda *a, **k: fake_labels)
    assert rc == 1


def test_metric_match_success(tmp_path, capsys):
    scores_src = (FIX / "scores_known.txt").read_text()
    scores = tmp_path / "s.txt"
    scores.write_text(scores_src)
    sha = hashlib.sha256(scores.read_bytes()).hexdigest()

    # Compute the EER the metric will produce, then pin it in the YAML.
    from speech_spoof_bench.metrics import get_metric
    parsed = {}
    for line in scores_src.splitlines():
        if line.strip():
            utt, s = line.split()
            parsed[utt] = float(s)
    labels = {"UTT_0000": 0, "UTT_0001": 1, "UTT_0002": 0, "UTT_0003": 1}
    expected = get_metric("eer_percent").fn(parsed, labels).value

    src = (FIX / "submissions" / "valid.yaml").read_text()
    data = yaml.safe_load(src)
    data["artifact"]["scores_sha256"] = sha
    data["scores"] = {
        "eer_percent": expected,
        "n_trials": 4,
        "n_skipped": 0,
    }
    p = tmp_path / "submission.yaml"
    p.write_text(yaml.safe_dump(data))

    with patch("speech_spoof_bench.reproduce.hf_fetch.download",
               return_value=(scores, sha)):
        rc = reproduce.run_scoring(p, label_stream=lambda *a, **k: labels)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "OK reproduced" in out
    assert "eer_percent" in out


def test_metric_mismatch(tmp_path):
    scores_src = (FIX / "scores_known.txt").read_text()
    scores = tmp_path / "s.txt"
    scores.write_text(scores_src)
    sha = hashlib.sha256(scores.read_bytes()).hexdigest()
    labels = {"UTT_0000": 0, "UTT_0001": 1, "UTT_0002": 0, "UTT_0003": 1}

    src = (FIX / "submissions" / "valid.yaml").read_text()
    data = yaml.safe_load(src)
    data["artifact"]["scores_sha256"] = sha
    data["scores"] = {"eer_percent": 0.0, "n_trials": 4, "n_skipped": 0}
    p = tmp_path / "submission.yaml"
    p.write_text(yaml.safe_dump(data))

    with patch("speech_spoof_bench.reproduce.hf_fetch.download",
               return_value=(scores, sha)):
        rc = reproduce.run_scoring(p, label_stream=lambda *a, **k: labels)
    assert rc == 1


def test_unknown_metric(tmp_path):
    scores_src = (FIX / "scores_known.txt").read_text()
    scores = tmp_path / "s.txt"
    scores.write_text(scores_src)
    sha = hashlib.sha256(scores.read_bytes()).hexdigest()
    labels = {"UTT_0000": 0, "UTT_0001": 1, "UTT_0002": 0, "UTT_0003": 1}

    src = (FIX / "submissions" / "valid.yaml").read_text()
    data = yaml.safe_load(src)
    data["artifact"]["scores_sha256"] = sha
    data["scores"] = {"made_up_metric": 1.23, "n_trials": 4, "n_skipped": 0}
    p = tmp_path / "submission.yaml"
    p.write_text(yaml.safe_dump(data))

    with patch("speech_spoof_bench.reproduce.hf_fetch.download",
               return_value=(scores, sha)):
        rc = reproduce.run_scoring(p, label_stream=lambda *a, **k: labels)
    assert rc == 1
