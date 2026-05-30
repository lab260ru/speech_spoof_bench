"""Tests for the nightly green store."""
from __future__ import annotations

from speech_spoof_bench.ci import green_store


def test_roundtrip_save_load(tmp_path):
    path = tmp_path / "green.json"
    store = {}
    green_store.record_green(store, "Org/Foo", "rb", "sha1", "rev1")
    green_store.save(store, path)
    loaded = green_store.load(path)
    assert loaded == store


def test_load_missing_returns_empty(tmp_path):
    assert green_store.load(tmp_path / "nope.json") == {}


def test_is_green_matches_on_all_three_fields():
    store = {}
    green_store.record_green(store, "Org/Foo", "rb", "sha1", "rev1")
    assert green_store.is_green(store, "Org/Foo", "rb", "sha1", "rev1") is True
    # Different sha / revision → not green.
    assert green_store.is_green(store, "Org/Foo", "rb", "sha2", "rev1") is False
    assert green_store.is_green(store, "Org/Foo", "rb", "sha1", "rev2") is False
    # Unknown submission → not green.
    assert green_store.is_green(store, "Org/Foo", "other", "sha1", "rev1") is False


def test_bench_version_change_invalidates(monkeypatch):
    store = {}
    green_store.record_green(store, "Org/Foo", "rb", "sha1", "rev1")
    # Simulate a new installed package version.
    monkeypatch.setattr(green_store, "_BENCH_VERSION", "9.9.9")
    assert green_store.is_green(store, "Org/Foo", "rb", "sha1", "rev1") is False


def test_load_corrupt_file_returns_empty(tmp_path):
    path = tmp_path / "green.json"
    path.write_text("{not valid json")
    assert green_store.load(path) == {}
