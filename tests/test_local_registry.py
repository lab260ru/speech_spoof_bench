"""Tests for local_registry (Phase 8, Slice 1)."""
from __future__ import annotations

from pathlib import Path

import pytest

from speech_spoof_bench import local_registry as lr


@pytest.fixture
def reg_path(tmp_path, monkeypatch):
    p = tmp_path / "local-datasets.yaml"
    monkeypatch.setattr(lr, "_registry_path", lambda: p)
    return p


@pytest.fixture
def dataset_dir(tmp_path):
    d = tmp_path / "LA"
    (d / "data").mkdir(parents=True)
    (d / "data" / "test-00000-of-00001.parquet").write_bytes(b"")
    (d / "eval.yaml").write_text("name: x\ntasks: [{split: test, metrics: [eer_percent]}]\n")
    return d


def test_load_missing_returns_empty(reg_path):
    assert lr.load() == {}


def test_set_then_load_roundtrip(reg_path, dataset_dir):
    lr.set("Org/Foo", dataset_dir)
    assert lr.load() == {"Org/Foo": str(dataset_dir)}


def test_set_rejects_missing_dir(reg_path, tmp_path):
    with pytest.raises(FileNotFoundError):
        lr.set("Org/Foo", tmp_path / "does-not-exist")


def test_set_rejects_dir_without_eval_yaml(reg_path, tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    with pytest.raises(ValueError, match="eval.yaml"):
        lr.set("Org/Foo", d)


def test_set_rejects_dir_without_parquet(reg_path, tmp_path):
    d = tmp_path / "bad"
    (d / "data").mkdir(parents=True)
    (d / "eval.yaml").write_text("name: x\n")
    with pytest.raises(ValueError, match="parquet"):
        lr.set("Org/Foo", d)


def test_unset_removes_entry(reg_path, dataset_dir):
    lr.set("Org/Foo", dataset_dir)
    lr.unset("Org/Foo")
    assert lr.load() == {}


def test_unset_unknown_id_is_noop(reg_path):
    lr.unset("Org/NotThere")  # must not raise


def test_lookup_returns_path_when_mapped(reg_path, dataset_dir):
    lr.set("Org/Foo", dataset_dir)
    assert lr.lookup("Org/Foo") == Path(str(dataset_dir))


def test_lookup_returns_none_when_unmapped(reg_path):
    assert lr.lookup("Org/Foo") is None


def test_lookup_raises_if_mapped_path_disappeared(reg_path, dataset_dir):
    lr.set("Org/Foo", dataset_dir)
    # Wipe the dataset dir after registration.
    for p in sorted(dataset_dir.rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    dataset_dir.rmdir()
    with pytest.raises(FileNotFoundError, match="registered as local"):
        lr.lookup("Org/Foo")


def test_set_overwrites_existing_entry(reg_path, dataset_dir, tmp_path):
    lr.set("Org/Foo", dataset_dir)
    other = tmp_path / "LA2"
    (other / "data").mkdir(parents=True)
    (other / "data" / "test-00000-of-00001.parquet").write_bytes(b"")
    (other / "eval.yaml").write_text("name: x\n")
    lr.set("Org/Foo", other)
    assert lr.load() == {"Org/Foo": str(other)}


def test_yaml_uses_list_schema(reg_path, dataset_dir):
    lr.set("Org/Foo", dataset_dir)
    import yaml
    raw = yaml.safe_load(reg_path.read_text())
    assert raw == {
        "schema_version": 1,
        "datasets": [{"id": "Org/Foo", "path": str(dataset_dir)}],
    }
