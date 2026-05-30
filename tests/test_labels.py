"""Tests for labels.py (emit_labels / load_labels_file)."""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from speech_spoof_bench import labels


def _write_shard(data_dir: Path, name: str, rows: list[tuple[str, int]]) -> None:
    """rows: list of (utterance_id, int_label)."""
    pq.write_table(
        pa.table({
            "path": [f"{u}.flac" for u, _ in rows],
            "audio": [b"" for _ in rows],
            "label": [lab for _, lab in rows],
            "notes": [json.dumps({"utterance_id": u}) for u, _ in rows],
        }),
        str(data_dir / name),
    )


def test_emit_labels_roundtrip(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_shard(data_dir, "test-00000-of-00002.parquet", [("u1", 0), ("u2", 1)])
    _write_shard(data_dir, "test-00001-of-00002.parquet", [("u3", 1)])

    out = labels.emit_labels(tmp_path)
    assert out == data_dir / "labels.parquet"
    assert out.is_file()
    assert labels.load_labels_file(out) == {"u1": 0, "u2": 1, "u3": 1}


def test_emit_labels_label_dtype_is_int8(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_shard(data_dir, "test-00000-of-00001.parquet", [("u1", 0), ("u2", 1)])
    out = labels.emit_labels(tmp_path)
    schema = pq.read_schema(str(out))
    assert schema.field("utterance_id").type == pa.string()
    assert schema.field("label").type == pa.int8()


def test_emit_labels_duplicate_uid_raises(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_shard(data_dir, "test-00000-of-00001.parquet", [("dup", 0), ("dup", 1)])
    with pytest.raises(ValueError, match="duplicate utterance_id"):
        labels.emit_labels(tmp_path)


def test_emit_labels_no_shards_raises(tmp_path):
    (tmp_path / "data").mkdir()
    with pytest.raises(FileNotFoundError):
        labels.emit_labels(tmp_path)
