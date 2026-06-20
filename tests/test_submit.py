"""Tests for submit._resolve_dataset_slug — the local-registry 404 fix (9.6)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from speech_spoof_bench import loader, submit
from speech_spoof_bench import local_registry as lr


def _fake_source(canonical_id: str, slug: str = "LA", split: str = "test"):
    return loader.DatasetSource(
        spec="x", display_name="X", slug=slug, canonical_id=canonical_id,
        metrics=["eer_percent"], split=split, is_local=True, local_path=None,
        revision=None,
    )


def _register(tmp_path, monkeypatch, dataset_id, dirname):
    reg = tmp_path / "local-datasets.yaml"
    monkeypatch.setattr(lr, "_registry_path", lambda: reg)
    dpath = tmp_path / dirname
    (dpath / "data").mkdir(parents=True)
    (dpath / "data" / "test-00000-of-00001.parquet").write_bytes(b"")
    (dpath / "eval.yaml").write_text(
        "name: x\ntasks: [{split: test, metrics: [eer_percent]}]\n"
    )
    lr.set(dataset_id, dpath)
    return dpath


def test_resolve_dataset_slug_local_path_recovers_canonical(monkeypatch, tmp_path):
    """A bare local path whose loader basename drops the org is recovered from the registry."""
    dpath = _register(
        tmp_path, monkeypatch,
        "SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA", "ASVspoof2021_LA",
    )
    # loader.resolve returns canonical_id == bare basename (the 404 trigger).
    monkeypatch.setattr(loader, "resolve", lambda spec, **kw: (_fake_source("ASVspoof2021_LA"), object()))
    api = MagicMock()
    api.repo_info.return_value = MagicMock(sha="deadbeef")

    canonical, slug, sha, split = submit._resolve_dataset_slug(str(dpath), api)

    assert canonical == "SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA"
    api.repo_info.assert_called_once_with(
        repo_id="SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA", repo_type="dataset"
    )
    assert sha == "deadbeef"


def test_resolve_dataset_slug_unknown_local_path_raises_clear_error(monkeypatch, tmp_path):
    """A bare path not in the registry gives an actionable error, not a 404."""
    reg = tmp_path / "local-datasets.yaml"
    monkeypatch.setattr(lr, "_registry_path", lambda: reg)  # empty registry
    monkeypatch.setattr(loader, "resolve", lambda spec, **kw: (_fake_source("ASVspoof2021_LA"), object()))
    api = MagicMock()

    with pytest.raises(ValueError, match="not a registered local dataset"):
        submit._resolve_dataset_slug("/tmp/whatever/ASVspoof2021_LA", api)
    api.repo_info.assert_not_called()


def test_resolve_dataset_slug_org_name_unchanged(monkeypatch):
    """The org/name happy path is untouched: no recovery, repo_info called directly."""
    monkeypatch.setattr(loader, "resolve", lambda spec, **kw: (_fake_source("Org/Name", slug="Name"), object()))
    api = MagicMock()
    api.repo_info.return_value = MagicMock(sha="cafe")

    canonical, slug, sha, split = submit._resolve_dataset_slug("Org/Name", api)

    assert canonical == "Org/Name"
    assert sha == "cafe"
    api.repo_info.assert_called_once_with(repo_id="Org/Name", repo_type="dataset")
