"""Tests for the dataset loader (local + HF dispatch)."""

from __future__ import annotations

from pathlib import Path

import pytest

# Import metric so eer_percent is registered (loader validates against registry).
import speech_spoof_bench.metrics.eer  # noqa: F401
from speech_spoof_bench.loader import DatasetSource, resolve


def test_local_dispatch(synth_local_dataset: Path):
    source, ds = resolve(str(synth_local_dataset), streaming=True)

    assert source.is_local is True
    assert source.local_path == synth_local_dataset
    assert source.slug == "SynthDataset_TEST"
    assert source.canonical_id == "SynthDataset_TEST"
    assert source.display_name == "Synth Dataset TEST"
    assert source.metrics == ["eer_percent"]
    assert source.split == "test"
    assert source.revision is None

    rows = list(ds)
    assert len(rows) == 4
    assert {r["label"] for r in rows} == {0, 1}


def test_local_dispatch_non_streaming(synth_local_dataset: Path):
    source, ds = resolve(str(synth_local_dataset), streaming=False)
    assert source.is_local is True
    assert len(ds) == 4


def test_local_missing_eval_yaml(tmp_path):
    bad = tmp_path / "no_eval"
    (bad / "data").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="eval.yaml"):
        resolve(str(bad))


def test_local_unknown_metric_in_eval_yaml(tmp_path):
    bad = tmp_path / "bad_metric"
    (bad / "data").mkdir(parents=True)
    # Drop a dummy parquet file so the parquet-glob check passes.
    (bad / "data" / "test-00000-of-00001.parquet").write_bytes(b"")
    (bad / "eval.yaml").write_text(
        "name: X\n"
        "tasks:\n"
        "  - id: t\n"
        "    config: default\n"
        "    split: test\n"
        "    metrics: [made_up_metric]\n"
    )
    with pytest.raises(KeyError, match="made_up_metric"):
        resolve(str(bad))


def test_local_no_parquet_shards(tmp_path):
    bad = tmp_path / "empty"
    (bad / "data").mkdir(parents=True)
    (bad / "eval.yaml").write_text(
        "name: X\n"
        "tasks:\n"
        "  - id: t\n"
        "    config: default\n"
        "    split: test\n"
        "    metrics: [eer_percent]\n"
    )
    with pytest.raises(FileNotFoundError, match="parquet"):
        resolve(str(bad))


def test_hf_dispatch_invokes_load_dataset(monkeypatch, tmp_path):
    """HF mode calls load_dataset and hf_hub_download with the repo id."""
    from speech_spoof_bench import loader as L

    fake_eval_path = tmp_path / "eval.yaml"
    fake_eval_path.write_text(
        "name: ASVspoof 2019 LA\n"
        "tasks:\n"
        "  - id: t\n"
        "    config: default\n"
        "    split: test\n"
        "    metrics: [eer_percent]\n"
    )

    called = {}

    def fake_load_dataset(*args, **kwargs):
        called["load_dataset"] = (args, kwargs)
        return ["row"]  # standin

    def fake_hf_hub_download(*, repo_id, filename, repo_type, **_):
        called["hf_hub_download"] = (repo_id, filename, repo_type)
        return str(fake_eval_path)

    monkeypatch.setattr(L, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(L, "hf_hub_download", fake_hf_hub_download)

    source, ds = resolve("SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA", streaming=True)
    assert source.is_local is False
    assert source.canonical_id == "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA"
    assert source.slug == "ASVspoof2019_LA"
    assert source.display_name == "ASVspoof 2019 LA"
    assert called["hf_hub_download"] == (
        "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA",
        "eval.yaml",
        "dataset",
    )


def test_bad_spec_neither_path_nor_repo_id():
    with pytest.raises(ValueError, match="not a directory"):
        resolve("just_a_word_no_slash_no_path")


def test_resolve_consults_local_registry(monkeypatch, tmp_path):
    """An HF id mapped in the registry resolves to the local directory."""
    from speech_spoof_bench import loader, local_registry as lr

    d = tmp_path / "LA"
    (d / "data").mkdir(parents=True)
    # minimal parquet shard with header — datasets refuses empty files
    import pyarrow as pa, pyarrow.parquet as pq
    pq.write_table(
        pa.table({"path": ["x"], "audio": [b""], "label": [0], "notes": ["{}"]}),
        d / "data" / "test-00000-of-00001.parquet",
    )
    (d / "eval.yaml").write_text(
        "name: t\ntasks: [{split: test, metrics: [eer_percent]}]\n"
    )
    monkeypatch.setattr(lr, "lookup", lambda did: d if did == "Org/Foo" else None)

    src, _ds = loader.resolve("Org/Foo", streaming=True)
    assert src.is_local is True
    assert src.local_path == d


def test_resolve_force_remote_bypasses_registry(monkeypatch, tmp_path):
    from speech_spoof_bench import loader, local_registry as lr

    monkeypatch.setattr(lr, "lookup", lambda did: tmp_path)  # should be ignored
    called = {}
    def fake_hf(repo, streaming):
        called["repo"] = repo
        # return a dummy source so the caller can finish
        return loader.DatasetSource(
            spec=repo, display_name=repo, slug=repo.split("/")[-1],
            canonical_id=repo, metrics=["eer_percent"], split="test",
            is_local=False, local_path=None, revision=None,
        ), None
    monkeypatch.setattr(loader, "_resolve_hf", fake_hf)

    loader.resolve("Org/Foo", streaming=True, force_remote=True)
    assert called["repo"] == "Org/Foo"
