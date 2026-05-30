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


def test_load_dataset_receives_columns_kwarg():
    """Guards the audio-not-fetched invariant.

    True column pushdown requires `columns=[...]` at load_dataset time.
    Post-construction select_columns() does NOT push down to the parquet
    reader. See implementation-notes for Phase 7a investigation.
    """
    captured: dict = {}

    def fake_load_dataset(*args, **kwargs):
        captured["kwargs"] = kwargs

        class FakeDS:
            def __iter__(self):
                return iter([
                    {"notes": '{"utterance_id":"a"}', "label": 0},
                    {"notes": '{"utterance_id":"b"}', "label": 1},
                ])

        return FakeDS()

    import speech_spoof_bench.reproduce as _rep
    _rep._LABEL_CACHE.clear()
    with patch("speech_spoof_bench.reproduce._download_labels_file", return_value=None):
        with patch("speech_spoof_bench.reproduce.load_dataset",
                   side_effect=fake_load_dataset):
            labels = reproduce._stream_labels("x/y", "test", "deadbeef")

    assert captured["kwargs"].get("columns") == ["notes", "label"]
    assert captured["kwargs"].get("streaming") is True
    assert captured["kwargs"].get("revision") == "deadbeef"
    assert captured["kwargs"].get("split") == "test"
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


def test_stream_labels_uses_local_registry(monkeypatch, tmp_path):
    """Falls back to local parquet shards when no labels.parquet is present."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from speech_spoof_bench import local_registry as lr

    d = tmp_path / "LA"
    (d / "data").mkdir(parents=True)
    pq.write_table(
        pa.table({
            "path": ["a", "b"],
            "audio": [b"", b""],
            "label": [0, 1],
            "notes": ['{"utterance_id":"u1"}', '{"utterance_id":"u2"}'],
        }),
        d / "data" / "test-00000-of-00001.parquet",
    )
    (d / "eval.yaml").write_text("name: t\n")
    monkeypatch.setattr(lr, "_registry_path", lambda: tmp_path / "reg.yaml")
    lr.set("Org/Foo", d)

    seen = {}
    def fake_load_dataset(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        # Return an iterable of two rows that mimic the parquet content.
        return iter([
            {"label": 0, "notes": '{"utterance_id":"u1"}'},
            {"label": 1, "notes": '{"utterance_id":"u2"}'},
        ])
    monkeypatch.setattr("speech_spoof_bench.reproduce.load_dataset", fake_load_dataset)
    reproduce._LABEL_CACHE.clear()

    labels = reproduce._stream_labels("Org/Foo", "test", "deadbee")
    assert labels == {"u1": 0, "u2": 1}
    # When mapped locally, first positional arg is "parquet", not the dataset id.
    assert seen["args"][0] == "parquet"


def test_stream_labels_force_remote_bypasses_registry(monkeypatch, tmp_path):
    from speech_spoof_bench import local_registry as lr

    d = tmp_path / "LA"
    (d / "data").mkdir(parents=True)
    (d / "data" / "test-00000-of-00001.parquet").write_bytes(b"")
    (d / "eval.yaml").write_text("name: t\n")
    monkeypatch.setattr(lr, "_registry_path", lambda: tmp_path / "reg.yaml")
    lr.set("Org/Foo", d)

    seen = {}
    def fake_load_dataset(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return iter([])
    monkeypatch.setattr("speech_spoof_bench.reproduce.load_dataset", fake_load_dataset)
    monkeypatch.setattr("speech_spoof_bench.reproduce._download_labels_file",
                        lambda did, rev: None)
    reproduce._LABEL_CACHE.clear()

    reproduce._stream_labels("Org/Foo", "test", "deadbee", force_remote=True)
    # force_remote bypasses the registry → HF code path → first positional is the dataset id.
    assert seen["args"][0] == "Org/Foo"


def test_stream_labels_reads_local_labels_file(monkeypatch, tmp_path):
    """A local mapped dir with labels.parquet is read directly (no shard stream)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from speech_spoof_bench import local_registry as lr

    d = tmp_path / "LA"
    (d / "data").mkdir(parents=True)
    (d / "eval.yaml").write_text("name: t\n")
    # A shard exists, but the labels file should win.
    (d / "data" / "test-00000-of-00001.parquet").write_bytes(b"")
    pq.write_table(
        pa.table({"utterance_id": pa.array(["u1", "u2"], pa.string()),
                  "label": pa.array([0, 1], pa.int8())}),
        d / "data" / "labels.parquet",
    )
    monkeypatch.setattr(lr, "_registry_path", lambda: tmp_path / "reg.yaml")
    lr.set("Org/Foo", d)

    def boom(*a, **k):
        raise AssertionError("load_dataset must not be called when labels.parquet exists")
    monkeypatch.setattr("speech_spoof_bench.reproduce.load_dataset", boom)
    reproduce._LABEL_CACHE.clear()

    labels = reproduce._stream_labels("Org/Foo", "test", "rev1")
    assert labels == {"u1": 0, "u2": 1}


def test_stream_labels_local_falls_back_to_shards_when_no_file(monkeypatch, tmp_path):
    from speech_spoof_bench import local_registry as lr
    d = tmp_path / "LA"
    (d / "data").mkdir(parents=True)
    (d / "eval.yaml").write_text("name: t\n")
    (d / "data" / "test-00000-of-00001.parquet").write_bytes(b"")
    monkeypatch.setattr(lr, "_registry_path", lambda: tmp_path / "reg.yaml")
    lr.set("Org/Foo", d)

    seen = {}
    def fake_load_dataset(*args, **kwargs):
        seen["args"] = args
        return iter([{"label": 0, "notes": '{"utterance_id":"u1"}'}])
    monkeypatch.setattr("speech_spoof_bench.reproduce.load_dataset", fake_load_dataset)
    reproduce._LABEL_CACHE.clear()

    labels = reproduce._stream_labels("Org/Foo", "test", "rev2")
    assert labels == {"u1": 0}
    assert seen["args"][0] == "parquet"


def test_stream_labels_remote_uses_labels_file(monkeypatch, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    lf = tmp_path / "labels.parquet"
    pq.write_table(
        pa.table({"utterance_id": pa.array(["a", "b"], pa.string()),
                  "label": pa.array([1, 0], pa.int8())}),
        lf,
    )
    monkeypatch.setattr("speech_spoof_bench.reproduce._download_labels_file",
                        lambda did, rev: lf)
    def boom(*a, **k):
        raise AssertionError("must not stream shards when labels file downloads")
    monkeypatch.setattr("speech_spoof_bench.reproduce.load_dataset", boom)
    reproduce._LABEL_CACHE.clear()

    labels = reproduce._stream_labels("x/y", "test", "deadbeef")
    assert labels == {"a": 1, "b": 0}


def test_stream_labels_remote_falls_back_when_file_absent(monkeypatch):
    monkeypatch.setattr("speech_spoof_bench.reproduce._download_labels_file",
                        lambda did, rev: None)
    def fake_load_dataset(*a, **k):
        return iter([{"label": 1, "notes": '{"utterance_id":"z"}'}])
    monkeypatch.setattr("speech_spoof_bench.reproduce.load_dataset", fake_load_dataset)
    reproduce._LABEL_CACHE.clear()

    labels = reproduce._stream_labels("x/y", "test", "rev404")
    assert labels == {"z": 1}


def test_stream_labels_process_cache_hit(monkeypatch):
    calls = {"n": 0}
    def fake_load_dataset(*a, **k):
        calls["n"] += 1
        return iter([{"label": 0, "notes": '{"utterance_id":"c"}'}])
    monkeypatch.setattr("speech_spoof_bench.reproduce._download_labels_file",
                        lambda did, rev: None)
    monkeypatch.setattr("speech_spoof_bench.reproduce.load_dataset", fake_load_dataset)
    reproduce._LABEL_CACHE.clear()

    a = reproduce._stream_labels("x/y", "test", "revC")
    b = reproduce._stream_labels("x/y", "test", "revC")
    assert a == b == {"c": 0}
    assert calls["n"] == 1  # second call served from the process cache


def test_stream_labels_force_shards_bypasses_file_and_cache(monkeypatch, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from speech_spoof_bench import local_registry as lr
    d = tmp_path / "LA"
    (d / "data").mkdir(parents=True)
    (d / "eval.yaml").write_text("name: t\n")
    (d / "data" / "test-00000-of-00001.parquet").write_bytes(b"")
    pq.write_table(
        pa.table({"utterance_id": pa.array(["u1"], pa.string()),
                  "label": pa.array([0], pa.int8())}),
        d / "data" / "labels.parquet",
    )
    monkeypatch.setattr(lr, "_registry_path", lambda: tmp_path / "reg.yaml")
    lr.set("Org/Foo", d)

    seen = {}
    def fake_load_dataset(*args, **kwargs):
        seen["args"] = args
        return iter([{"label": 1, "notes": '{"utterance_id":"shard_only"}'}])
    monkeypatch.setattr("speech_spoof_bench.reproduce.load_dataset", fake_load_dataset)
    reproduce._LABEL_CACHE.clear()

    labels = reproduce._stream_labels("Org/Foo", "test", "rev", force_shards=True)
    assert labels == {"shard_only": 1}   # came from shards, not labels.parquet
    assert seen["args"][0] == "parquet"
