"""Derive and read the per-dataset ``data/labels.parquet`` fast-path artifact.

``emit_labels`` reads the canonical shards' ``notes`` + ``label`` columns only
(no audio) and writes ``data/labels.parquet`` with two typed columns:
``utterance_id`` (string) and ``label`` (int8, 0=bonafide 1=spoof). The shards
remain the source of truth, so emit derives the map from them, asserts the
written file matches, and refuses to leave an inconsistent file behind.

``load_labels_file`` reads the file back to ``{utterance_id: int}`` for
``reproduce._stream_labels`` — one columnar read instead of an 80-shard stream.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

LABELS_FILENAME = "labels.parquet"


def _shard_paths(data_dir: Path) -> list[str]:
    shards = sorted(glob.glob(str(data_dir / "test-*.parquet")))
    if not shards:
        raise FileNotFoundError(f"{data_dir}/test-*.parquet not found")
    return shards


def emit_labels(dataset_dir: Path | str) -> Path:
    """Derive ``<dataset_dir>/data/labels.parquet`` from the shards. Returns it."""
    data_dir = Path(dataset_dir) / "data"
    uids: list[str] = []
    label_vals: list[int] = []
    seen: set[str] = set()
    for shard in _shard_paths(data_dir):
        t = pq.read_table(shard, columns=["notes", "label"])
        notes = t.column("notes").to_pylist()
        labs = t.column("label").to_pylist()
        for note, lab in zip(notes, labs):
            uid = json.loads(note)["utterance_id"]
            if uid in seen:
                raise ValueError(f"duplicate utterance_id in shards: {uid!r}")
            seen.add(uid)
            uids.append(uid)
            label_vals.append(int(lab))

    out_path = data_dir / LABELS_FILENAME
    pq.write_table(
        pa.table({
            "utterance_id": pa.array(uids, pa.string()),
            "label": pa.array(label_vals, pa.int8()),
        }),
        str(out_path),
    )

    # Consistency assert: the written file must reproduce the shard-derived map.
    if load_labels_file(out_path) != dict(zip(uids, label_vals)):
        out_path.unlink(missing_ok=True)
        raise AssertionError("labels.parquet does not match shards after write")
    return out_path


def load_labels_file(path: Path | str) -> dict[str, int]:
    """Read a ``labels.parquet`` into ``{utterance_id: int_label}``."""
    t = pq.read_table(str(path), columns=["utterance_id", "label"])
    uids = t.column("utterance_id").to_pylist()
    labs = t.column("label").to_pylist()
    return {u: int(lab) for u, lab in zip(uids, labs)}
