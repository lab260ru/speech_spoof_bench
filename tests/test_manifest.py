"""Tests for speech_spoof_bench.manifest."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from jsonschema import ValidationError

from speech_spoof_bench import manifest as mf


VALID = {
    "ranking_version": "v1",
    "schema_version": 1,
    "metrics_in_use": ["eer_percent"],
    "tiers": [
        {"name": "gold", "min_coverage": 1.0},
        {"name": "silver", "min_coverage": 0.5},
        {"name": "bronze", "min_coverage": 0.0},
    ],
    "core_set": [
        {"id": "Org/Dataset_A", "revision": "9b2040e8c57749dcd9a4f16ad61b4f47626b89ec"}
    ],
    "extended": [
        {"id": "Org/Dataset_B", "revision": "deadbee"}
    ],
}


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def test_load_manifest_accepts_valid(tmp_path):
    out = mf.load_manifest(_write(tmp_path, VALID))
    assert out["ranking_version"] == "v1"
    assert out["schema_version"] == 1


@pytest.mark.parametrize("mutator,reason", [
    (lambda d: d.pop("tiers"), "missing tiers"),
    (lambda d: d.pop("core_set"), "missing core_set"),
    (lambda d: d.update({"extra_top_level_key": 1}), "extra key"),
    (lambda d: d["core_set"].clear(), "empty core_set"),
    (lambda d: d["core_set"][0].update({"revision": "not-hex"}), "bad revision"),
    (lambda d: d["core_set"][0].update({"id": "no-slash"}), "bad id"),
    (lambda d: d["tiers"][0].update({"min_coverage": 1.5}), "min_coverage > 1"),
    (lambda d: d.update({"schema_version": 2}), "wrong schema_version"),
    (lambda d: d["metrics_in_use"].clear(), "empty metrics"),
])
def test_load_manifest_rejects_invalid(tmp_path, mutator, reason):
    import copy
    bad = copy.deepcopy(VALID)
    mutator(bad)
    with pytest.raises(ValidationError):
        mf.load_manifest(_write(tmp_path, bad))


def test_core_dataset_ids(tmp_path):
    m = mf.load_manifest(_write(tmp_path, VALID))
    assert mf.core_dataset_ids(m) == ["Org/Dataset_A"]


def test_all_dataset_ids_core_then_extended(tmp_path):
    m = mf.load_manifest(_write(tmp_path, VALID))
    assert mf.all_dataset_ids(m) == ["Org/Dataset_A", "Org/Dataset_B"]


def test_revision_for_known(tmp_path):
    m = mf.load_manifest(_write(tmp_path, VALID))
    assert mf.revision_for(m, "Org/Dataset_A").startswith("9b2040e8")
    assert mf.revision_for(m, "Org/Dataset_B") == "deadbee"


def test_revision_for_unknown_returns_none(tmp_path):
    m = mf.load_manifest(_write(tmp_path, VALID))
    assert mf.revision_for(m, "Org/Nope") is None


def test_fetch_manifest_uses_hf_hub(monkeypatch, tmp_path):
    """fetch_manifest delegates to hf_hub_download with the expected repo coords."""
    fake = _write(tmp_path, VALID)
    calls = {}

    def fake_download(*, repo_id, repo_type, filename):
        calls["repo_id"] = repo_id
        calls["repo_type"] = repo_type
        calls["filename"] = filename
        return str(fake)

    monkeypatch.setattr(mf, "hf_hub_download", fake_download)
    out = mf.fetch_manifest()
    assert out["ranking_version"] == "v1"
    assert calls == {
        "repo_id": "SpeechAntiSpoofingBenchmarks/arena-manifest",
        "repo_type": "dataset",
        "filename": "manifest.yaml",
    }
