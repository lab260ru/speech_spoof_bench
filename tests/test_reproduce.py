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
