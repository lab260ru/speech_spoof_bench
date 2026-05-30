"""Tests for the `emit-labels` CLI subcommand."""
from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from speech_spoof_bench import cli, labels


def test_emit_labels_cli_writes_file(tmp_path, capsys):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pq.write_table(
        pa.table({
            "path": ["u1.flac", "u2.flac"],
            "audio": [b"", b""],
            "label": [0, 1],
            "notes": [json.dumps({"utterance_id": "u1"}),
                      json.dumps({"utterance_id": "u2"})],
        }),
        str(data_dir / "test-00000-of-00001.parquet"),
    )
    parser = cli.build_parser()
    args = parser.parse_args(["emit-labels", str(tmp_path)])
    rc = args.func(args)
    assert rc == 0
    out = data_dir / "labels.parquet"
    assert out.is_file()
    assert labels.load_labels_file(out) == {"u1": 0, "u2": 1}


def test_reproduce_parser_accepts_force_shards():
    parser = cli.build_parser()
    args = parser.parse_args(["reproduce", "sub.yaml", "--scoring", "--force-shards"])
    assert args.force_shards is True
