"""Reproduce a spoof-only (srr_complement) submission via the DeepVoice-calibrated threshold."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import yaml

from speech_spoof_bench import reproduce

FIX = Path(__file__).parent / "fixtures"

# Four spoof utterances; at t*=0 two are below (rejected) and two above
# => SRR = 0.5 => srr_complement = 50.0.
_SCORES_TXT = "u0 -1.0\nu1 -1.0\nu2 1.0\nu3 1.0\n"
_SPOOF_LABELS = {"u0": 1, "u1": 1, "u2": 1, "u3": 1}


def _srr_submission(tmp_path, scores_sha, *, with_calibration: bool):
    src = (FIX / "submissions" / "valid.yaml").read_text()
    data = yaml.safe_load(src)
    data["schema_version"] = 5
    data["dataset"]["id"] = "lab260/LRLspoof"
    data["artifact"]["scores_sha256"] = scores_sha
    data["scores"] = {"srr_complement": 50.0, "n_trials": 4, "n_skipped": 0}
    if with_calibration:
        data["calibration"] = {
            "source_dataset": "SpeechAntiSpoofingBenchmarks/DeepVoice",
            "threshold": 0.0,
        }
    p = tmp_path / "submission.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_reproduce_srr_with_calibration_succeeds(tmp_path, capsys):
    scores = tmp_path / "s.txt"
    scores.write_text(_SCORES_TXT)
    sha = hashlib.sha256(scores.read_bytes()).hexdigest()
    yaml_path = _srr_submission(tmp_path, sha, with_calibration=True)
    with patch("speech_spoof_bench.reproduce.hf_fetch.download",
               return_value=(scores, sha)):
        rc = reproduce.run_scoring(
            yaml_path, label_stream=lambda *a, **k: _SPOOF_LABELS
        )
    out = capsys.readouterr().out
    assert rc == 0
    assert "srr_complement" in out


def test_reproduce_srr_without_calibration_fails_clean(tmp_path):
    # No calibration block => srr_complement has no threshold => the metric
    # raises, and run_scoring must FAIL cleanly (return 1), not crash.
    scores = tmp_path / "s.txt"
    scores.write_text(_SCORES_TXT)
    sha = hashlib.sha256(scores.read_bytes()).hexdigest()
    yaml_path = _srr_submission(tmp_path, sha, with_calibration=False)
    with patch("speech_spoof_bench.reproduce.hf_fetch.download",
               return_value=(scores, sha)):
        rc = reproduce.run_scoring(
            yaml_path, label_stream=lambda *a, **k: _SPOOF_LABELS
        )
    assert rc == 1
