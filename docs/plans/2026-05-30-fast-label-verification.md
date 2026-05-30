# Fast Label Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make label-only reproduction and nightly revalidation fast by shipping a tiny per-dataset `data/labels.parquet`, memoizing labels in-process, and skipping unchanged nightly submissions.

**Architecture:** Three layers, all graceful and additive. (1) A `labels.py` module derives/reads `data/labels.parquet` from the canonical shards (no audio); `reproduce._stream_labels` tries this one-request file before falling back to the 80-shard stream. (2) A process-level dict memoizes `{uid:label}` per `(dataset_id, revision)`. (3) `nightly` skips a submission entirely when `(scores_sha256, revision, installed bench_version)` is unchanged since the last green run, persisting that state via a JSON file cached by GitHub Actions.

**Tech Stack:** Python 3.11, `pyarrow`, `datasets`, `huggingface_hub`, `argparse` CLI, `pytest`, GitHub Actions.

**Spec:** `docs/specs/2026-05-30-fast-label-verification-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/speech_spoof_bench/labels.py` (new) | Derive `data/labels.parquet` from shards (`emit_labels`); read it back (`load_labels_file`). Shards stay source of truth. |
| `src/speech_spoof_bench/reproduce.py` (modify) | Labels-file fast path + process cache in `_stream_labels`; `force_shards` escape hatch threaded through `run_scoring`. |
| `src/speech_spoof_bench/ci/green_store.py` (new) | Load/save the nightly "last green verified" JSON; key by `(scores_sha256, revision, bench_version)`. |
| `src/speech_spoof_bench/ci/nightly.py` (modify) | Skip-unchanged via green store; `--full` override; skipped/verified logging. |
| `src/speech_spoof_bench/cli.py` (modify) | `emit-labels` subcommand; `reproduce --force-shards`; `ci nightly-revalidate --full`. |
| `.github/workflows/nightly-revalidate.yml` (modify) | `actions/cache` for the green store; weekly `--full` schedule. |
| `benchmarks/{ASVspoof2021_DF,ASVspoof2019_LA}/build_parquet.py` (modify) | Call `labels.emit_labels` at end of a full build. |
| `src/speech_spoof_bench/__init__.py`, `pyproject.toml` (modify) | Version bump `0.1.1 → 0.2.0`. |
| Tests + docs | Cover every behavior; document the file, command, cache, and nightly skip. |

---

## Task 1: `labels.py` — derive and read `labels.parquet`

**Files:**
- Create: `src/speech_spoof_bench/labels.py`
- Test: `tests/test_labels.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_labels.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_labels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'speech_spoof_bench.labels'`.

- [ ] **Step 3: Write the implementation**

Create `src/speech_spoof_bench/labels.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_labels.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/speech_spoof_bench/labels.py tests/test_labels.py
git commit -m "feat: labels.py — derive/read data/labels.parquet from shards"
```

---

## Task 2: `reproduce` — labels-file fast path + process cache + `force_shards`

**Files:**
- Modify: `src/speech_spoof_bench/reproduce.py:34-99`
- Test: `tests/test_reproduce.py` (add new tests; update two existing remote-path tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reproduce.py`:

```python
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
```

Also **update the two existing remote-path tests** so they don't attempt a real download. In `test_load_dataset_receives_columns_kwarg` and `test_stream_labels_force_remote_bypasses_registry`, add this line right after the `with patch(... "load_dataset" ...)` / `monkeypatch.setattr(... "load_dataset" ...)` setup and before calling `_stream_labels` (so the labels-file lookup returns None and the shard path runs as the test expects):

In `test_load_dataset_receives_columns_kwarg` (currently uses `with patch(...)`), convert to also patch the downloader — add inside the `with` block a nested patch, or prepend:

```python
    # New: ensure the remote labels-file fast path is a no-op for this test.
    import speech_spoof_bench.reproduce as _rep
    _rep._LABEL_CACHE.clear()
    with patch("speech_spoof_bench.reproduce._download_labels_file", return_value=None):
        with patch("speech_spoof_bench.reproduce.load_dataset",
                   side_effect=fake_load_dataset):
            labels = reproduce._stream_labels("x/y", "test", "deadbeef")
```

In `test_stream_labels_force_remote_bypasses_registry`, add before the call:

```python
    monkeypatch.setattr("speech_spoof_bench.reproduce._download_labels_file",
                        lambda did, rev: None)
    reproduce._LABEL_CACHE.clear()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reproduce.py -v`
Expected: the new tests FAIL with `AttributeError: ... has no attribute '_LABEL_CACHE'` / `_download_labels_file`.

- [ ] **Step 3: Write the implementation**

In `src/speech_spoof_bench/reproduce.py`, replace the whole `_stream_labels` function (lines 34–83) with the following, and add the module-level cache + helpers. Keep the existing imports; add nothing at module top except the cache dict:

```python
# Process-level memo of immutable labels, keyed by (dataset_id, revision).
_LABEL_CACHE: dict[tuple[str, str], dict[str, int]] = {}


def _download_labels_file(dataset_id: str, revision: str):
    """Download ``data/labels.parquet`` from the pinned dataset revision.

    Returns the local file path, or ``None`` if the file is absent (older
    datasets) or any fetch error occurs — the caller then streams shards.
    Never raises: the labels file is a pure optimization.
    """
    from pathlib import Path as _Path
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=dataset_id,
            filename="data/labels.parquet",
            repo_type="dataset",
            revision=revision,
        )
        return _Path(local)
    except Exception:
        return None


def _stream_labels_from_shards(dataset_id, split, revision, *, mapped):
    """Original behavior: stream (notes, label) from local or remote shards."""
    if mapped is not None:
        import glob
        shards = sorted(glob.glob(str(mapped / "data" / "test-*.parquet")))
        if not shards:
            raise FileNotFoundError(
                f"{mapped}/data/test-*.parquet not found for {dataset_id}"
            )
        ds = load_dataset(
            "parquet", data_files={"train": shards}, split="train",
            streaming=True, columns=["notes", "label"],
        )
    else:
        ds = load_dataset(
            dataset_id, split=split, streaming=True, revision=revision,
            columns=["notes", "label"],
        )
    labels: dict[str, int] = {}
    for row in ds:
        note = json.loads(row["notes"])
        labels[note["utterance_id"]] = int(row["label"])
    return labels


def _stream_labels(
    dataset_id: str, split: str, revision: str, *,
    force_remote: bool = False, force_shards: bool = False,
) -> dict[str, int]:
    """Return ``{utterance_id: int_label}`` for a pinned dataset revision.

    Resolution order (shards stay authoritative; the labels file and cache are
    pure optimizations and never cause a failure when absent):
      1. process cache (unless ``force_shards``)
      2. ``data/labels.parquet`` — local mapped copy, or one HF download
      3. stream ``data/test-*.parquet`` shards (today's behavior)

    ``force_remote`` ignores the local-dataset registry. ``force_shards``
    bypasses the labels file and the cache to verify against shards directly.
    """
    from . import local_registry, labels as labels_mod

    cache_key = (dataset_id, revision)
    if not force_shards and cache_key in _LABEL_CACHE:
        return _LABEL_CACHE[cache_key]

    mapped = None if force_remote else local_registry.lookup(dataset_id)

    result: dict[str, int] | None = None
    if not force_shards:
        if mapped is not None:
            lf = mapped / "data" / labels_mod.LABELS_FILENAME
            if lf.is_file():
                result = labels_mod.load_labels_file(lf)
        else:
            lf = _download_labels_file(dataset_id, revision)
            if lf is not None:
                result = labels_mod.load_labels_file(lf)

    if result is None:
        result = _stream_labels_from_shards(
            dataset_id, split, revision, mapped=mapped
        )

    _LABEL_CACHE[cache_key] = result
    return result
```

Then thread `force_shards` through `run_scoring`. Change its signature (line 86) and the default `label_stream` (lines 97–99):

```python
def run_scoring(
    yaml_path: Path | str,
    *,
    tolerance: float = 1e-6,
    force_remote: bool = False,
    force_shards: bool = False,
    label_stream=None,
) -> int:
    """Run --scoring reproduction. Returns exit code (0 success, 1 fail).

    ``label_stream`` is injectable for tests. Defaults to _stream_labels.
    """
    if label_stream is None:
        def label_stream(did, split, rev):
            return _stream_labels(
                did, split, rev,
                force_remote=force_remote, force_shards=force_shards,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reproduce.py -v`
Expected: PASS (all existing + new tests).

- [ ] **Step 5: Commit**

```bash
git add src/speech_spoof_bench/reproduce.py tests/test_reproduce.py
git commit -m "feat: reproduce labels.parquet fast path + process cache + force_shards"
```

---

## Task 3: CLI — `emit-labels` command and `reproduce --force-shards`

**Files:**
- Modify: `src/speech_spoof_bench/cli.py:81-87` (`_cmd_reproduce`), `:268-286` (reproduce parser); add `emit-labels` near `scaffold-dataset`.
- Test: `tests/test_cli_reproduce.py` (add); `tests/test_cli_emit_labels.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli_emit_labels.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli_emit_labels.py -v`
Expected: FAIL — `argument cmd: invalid choice: 'emit-labels'` and `AttributeError: 'Namespace' object has no attribute 'force_shards'`.

- [ ] **Step 3: Write the implementation**

In `cli.py`, add the command handler near `_cmd_scaffold_dataset` (after line 116):

```python
def _cmd_emit_labels(args: argparse.Namespace) -> int:
    from . import labels
    out = labels.emit_labels(args.dataset_dir)
    print(f"wrote {out}")
    return 0
```

Update `_cmd_reproduce` (lines 81–87) to pass `force_shards`:

```python
def _cmd_reproduce(args: argparse.Namespace) -> int:
    if args.inference:
        raise NotImplementedError("reproduce --inference lands in Phase 7b/8")
    from . import reproduce
    return reproduce.run_scoring(
        args.path, tolerance=args.tolerance, force_remote=args.no_local,
        force_shards=args.force_shards,
    )
```

Add the `--force-shards` argument to the `reproduce` parser (after line 285, before `rp.set_defaults`):

```python
    rp.add_argument(
        "--force-shards",
        action="store_true",
        help="bypass data/labels.parquet; verify against shards directly",
    )
```

Register the `emit-labels` parser (after the `scaffold-dataset` block ending at line 266):

```python
    el = sub.add_parser(
        "emit-labels",
        help="derive data/labels.parquet from a dataset's shards (no audio)",
    )
    el.add_argument("dataset_dir", type=Path,
                    help="local dataset repo dir containing data/test-*.parquet")
    el.set_defaults(func=_cmd_emit_labels)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli_emit_labels.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/speech_spoof_bench/cli.py tests/test_cli_emit_labels.py
git commit -m "feat: emit-labels CLI command + reproduce --force-shards flag"
```

---

## Task 4: `ci/green_store.py` — nightly last-green state

**Files:**
- Create: `src/speech_spoof_bench/ci/green_store.py`
- Test: `tests/ci/test_green_store.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/ci/test_green_store.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/ci/test_green_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'speech_spoof_bench.ci.green_store'`.

- [ ] **Step 3: Write the implementation**

Create `src/speech_spoof_bench/ci/green_store.py`:

```python
"""Persistence for nightly skip-unchanged.

A submission's reproduction result is a pure function of
``(scores_sha256, dataset.revision, installed bench_version)``. When all three
are unchanged since the last green verification, the result cannot have moved,
so nightly skips it. The store is a plain ``{submission_id: entry_key}`` JSON
dict persisted via ``actions/cache``; losing it only costs a safe re-verify.
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import __version__ as _BENCH_VERSION

DEFAULT_STORE_PATH = Path(".nightly-green.json")


def submission_id(dataset_id: str, slug: str) -> str:
    return f"{dataset_id}/{slug}"


def _entry_key(scores_sha256: str, revision: str) -> str:
    return f"{scores_sha256}|{revision}|{_BENCH_VERSION}"


def load(path: Path | str = DEFAULT_STORE_PATH) -> dict[str, str]:
    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text())


def save(store: dict[str, str], path: Path | str = DEFAULT_STORE_PATH) -> None:
    Path(path).write_text(json.dumps(store, indent=2, sort_keys=True))


def is_green(store: dict[str, str], dataset_id: str, slug: str,
             scores_sha256: str, revision: str) -> bool:
    return store.get(submission_id(dataset_id, slug)) == _entry_key(
        scores_sha256, revision
    )


def record_green(store: dict[str, str], dataset_id: str, slug: str,
                 scores_sha256: str, revision: str) -> None:
    store[submission_id(dataset_id, slug)] = _entry_key(scores_sha256, revision)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/ci/test_green_store.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/speech_spoof_bench/ci/green_store.py tests/ci/test_green_store.py
git commit -m "feat: nightly green store (skip-unchanged state)"
```

---

## Task 5: `nightly` — skip-unchanged + `--full` + logging

**Files:**
- Modify: `src/speech_spoof_bench/ci/nightly.py:40-72,127-137`
- Modify: `src/speech_spoof_bench/cli.py:163-165` (`_cmd_ci_nightly`), `:321-325` (nr parser)
- Test: `tests/ci/test_nightly.py` (rewrite `test_collect_failures_*`, add skip tests)

- [ ] **Step 1: Write the failing tests**

In `tests/ci/test_nightly.py`, **replace** `test_collect_failures_calls_reproduce_per_submission` with the following, and add the new tests below it:

```python
def _sub(slug="rb", sha="sha1", rev="rev1"):
    return {
        "system": {"slug": slug},
        "artifact": {"scores_sha256": sha},
        "dataset": {"revision": rev},
    }


def _wire(monkeypatch, subs, check_result):
    """subs: {path: submission_dict}; check_result: callable(did, data)->Failure|None."""
    monkeypatch.setattr(nightly, "_fetch_manifest",
                        lambda: {"core_set": [{"id": "Org/Foo"}], "extended": []})
    monkeypatch.setattr(nightly, "_list_submission_files",
                        lambda did, **_: list(subs.keys()))
    monkeypatch.setattr(nightly, "fetch_submission", lambda did, p: subs[p])
    monkeypatch.setattr(nightly, "_check_submission_data", check_result)


def test_collect_failures_verifies_and_records_green(monkeypatch, tmp_path):
    from speech_spoof_bench.ci import green_store
    store_path = tmp_path / "green.json"
    monkeypatch.setattr(green_store, "DEFAULT_STORE_PATH", store_path)
    subs = {"submissions/a.yaml": _sub("a"), "submissions/b.yaml": _sub("b")}
    called = []
    def check(did, data):
        called.append(data["system"]["slug"])
        return nightly.Failure(did, "b", "EER drift") if data["system"]["slug"] == "b" else None
    _wire(monkeypatch, subs, check)

    failures = nightly.collect_failures()
    assert set(called) == {"a", "b"}
    assert failures == [nightly.Failure("Org/Foo", "b", "EER drift")]
    # The passing submission 'a' was recorded green; 'b' (failed) was not.
    saved = green_store.load(store_path)
    assert green_store.is_green(saved, "Org/Foo", "a", "sha1", "rev1") is True
    assert green_store.is_green(saved, "Org/Foo", "b", "sha1", "rev1") is False


def test_collect_failures_skips_green_submission(monkeypatch, tmp_path):
    from speech_spoof_bench.ci import green_store
    store_path = tmp_path / "green.json"
    monkeypatch.setattr(green_store, "DEFAULT_STORE_PATH", store_path)
    # Pre-seed 'a' as green.
    seed = {}
    green_store.record_green(seed, "Org/Foo", "a", "sha1", "rev1")
    green_store.save(seed, store_path)

    subs = {"submissions/a.yaml": _sub("a")}
    called = []
    _wire(monkeypatch, subs, lambda did, data: called.append(data) or None)

    failures = nightly.collect_failures()
    assert failures == []
    assert called == []  # skipped — reproduce never ran


def test_collect_failures_full_ignores_store(monkeypatch, tmp_path):
    from speech_spoof_bench.ci import green_store
    store_path = tmp_path / "green.json"
    monkeypatch.setattr(green_store, "DEFAULT_STORE_PATH", store_path)
    seed = {}
    green_store.record_green(seed, "Org/Foo", "a", "sha1", "rev1")
    green_store.save(seed, store_path)

    subs = {"submissions/a.yaml": _sub("a")}
    called = []
    _wire(monkeypatch, subs, lambda did, data: called.append(data) or None)

    nightly.collect_failures(full=True)
    assert len(called) == 1  # --full re-verifies despite the green entry
```

Note: the `_check_submission`/`fetch_submission` names must be importable on the `nightly` module for monkeypatching — Step 3 imports `fetch_submission` at module scope.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/ci/test_nightly.py -v`
Expected: FAIL — `collect_failures()` doesn't accept `full`, `_check_submission_data` / module-level `fetch_submission` don't exist.

- [ ] **Step 3: Write the implementation**

In `src/speech_spoof_bench/ci/nightly.py`, the import block at the top already has
`from ..submission import list_submission_files, fetch_submission`. Add the green store import:

```python
from . import green_store
```

Replace `_check_submission` (lines 40–55) with a data-based helper plus a thin wrapper:

```python
def _check_submission_data(dataset_id: str, data: dict) -> Failure | None:
    """Run reproduce --scoring on an already-fetched submission dict."""
    import tempfile
    import yaml as _yaml
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        _yaml.safe_dump(data, fh)
        local = fh.name
    rc = reproduce.run_scoring(local, tolerance=1e-6)
    if rc != 0:
        return Failure(dataset_id, data["system"]["slug"],
                       "reproduce --scoring failed (see job log)")
    return None


def _check_submission(dataset_id: str, path: str) -> Failure | None:
    try:
        data = fetch_submission(dataset_id, path)
    except submission.SubmissionValidationError as e:
        return Failure(dataset_id, path.rsplit("/", 1)[-1].removesuffix(".yaml"),
                       f"schema: {e}")
    return _check_submission_data(dataset_id, data)
```

Replace `collect_failures` (lines 58–72) with the skip-aware version:

```python
def collect_failures(*, full: bool = False) -> list[Failure]:
    failures: list[Failure] = []
    store = green_store.load()
    n_skipped = n_verified = 0
    m = _fetch_manifest()
    for entry in m.get("core_set", []) + m.get("extended", []):
        did = entry["id"]
        try:
            paths = _list_submission_files(did)
        except Exception as exc:  # noqa: BLE001
            failures.append(Failure(did, "<list>", f"list failed: {exc}"))
            continue
        for p in paths:
            try:
                data = fetch_submission(did, p)
            except submission.SubmissionValidationError as e:
                failures.append(Failure(
                    did, p.rsplit("/", 1)[-1].removesuffix(".yaml"), f"schema: {e}"))
                continue
            slug = data["system"]["slug"]
            sha = data["artifact"]["scores_sha256"]
            rev = data["dataset"]["revision"]
            if not full and green_store.is_green(store, did, slug, sha, rev):
                n_skipped += 1
                logger.info("nightly skip (unchanged/green): %s/%s", did, slug)
                continue
            n_verified += 1
            f = _check_submission_data(did, data)
            if f is not None:
                failures.append(f)
            else:
                green_store.record_green(store, did, slug, sha, rev)
    green_store.save(store)
    logger.info("nightly: %d verified, %d skipped", n_verified, n_skipped)
    return failures
```

Update `run` (lines 127–137) to accept and forward `full`:

```python
def run(*, open_issues: bool, full: bool = False) -> int:
    failures = collect_failures(full=full)
    for f in failures:
        logger.warning("nightly failure: %s/%s — %s", f.dataset_id, f.slug, f.reason)
    if open_issues:
        try:
            manage_issues(failures=failures)
        except Exception as exc:  # noqa: BLE001
            logger.warning("issue management failed: %s", exc)
    return 0 if not failures else 1
```

In `cli.py`, update `_cmd_ci_nightly` (lines 163–165):

```python
def _cmd_ci_nightly(args: argparse.Namespace) -> int:
    from .ci import nightly
    return nightly.run(open_issues=args.open_issues, full=args.full)
```

Add the `--full` flag to the `nightly-revalidate` parser (after line 324, before `nr.set_defaults`):

```python
    nr.add_argument("--full", action="store_true",
        help="re-verify every submission, ignoring the green store")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/ci/test_nightly.py -v`
Expected: PASS (existing `manage_issues` tests + new collect tests).

- [ ] **Step 5: Commit**

```bash
git add src/speech_spoof_bench/ci/nightly.py src/speech_spoof_bench/cli.py tests/ci/test_nightly.py
git commit -m "feat: nightly skip-unchanged via green store + --full override"
```

---

## Task 6: Workflow — cache the green store + weekly full sweep

**Files:**
- Modify: `.github/workflows/nightly-revalidate.yml`

- [ ] **Step 1: Rewrite the workflow**

Replace the entire contents of `.github/workflows/nightly-revalidate.yml` with:

```yaml
name: nightly-revalidate

on:
  schedule:
    - cron: "0 6 * * *"   # 06:00 UTC daily — incremental (skip-unchanged)
    - cron: "0 5 * * 0"   # 05:00 UTC Sunday — full sweep (catches artifact drift)
  workflow_dispatch:
    inputs:
      full:
        description: "Re-verify every submission (ignore green store)"
        type: boolean
        default: false

jobs:
  walk:
    runs-on: ubuntu-latest
    permissions:
      issues: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e .
      - name: Restore nightly green store
        uses: actions/cache@v4
        with:
          path: .nightly-green.json
          key: nightly-green-${{ github.run_id }}
          restore-keys: |
            nightly-green-
      - name: Determine mode
        id: mode
        run: |
          if [ "${{ github.event.schedule }}" = "0 5 * * 0" ] || [ "${{ inputs.full }}" = "true" ]; then
            echo "flag=--full" >> "$GITHUB_OUTPUT"
          else
            echo "flag=" >> "$GITHUB_OUTPUT"
          fi
      - name: Walk submissions
        env:
          HF_BOT_TOKEN: ${{ secrets.HF_BOT_TOKEN }}
          GH_TOKEN:     ${{ secrets.GITHUB_TOKEN }}
        run: speech-spoof-bench ci nightly-revalidate --open-issues ${{ steps.mode.outputs.flag }}
```

- [ ] **Step 2: Validate the YAML parses**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/nightly-revalidate.yml')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/nightly-revalidate.yml
git commit -m "ci: cache nightly green store via actions/cache + weekly full sweep"
```

> The rolling cache (`key` per `run_id` + `restore-keys: nightly-green-`) restores the most recent prior store and saves a fresh copy each run. Correctness does not depend on the cache: a miss just re-verifies. The `_BENCH_VERSION` embedded in each entry key handles release-driven invalidation.

---

## Task 7: Builders emit `labels.parquet`

**Files:**
- Modify: `benchmarks/ASVspoof2021_DF/build_parquet.py:267-268`
- Modify: `benchmarks/ASVspoof2019_LA/build_parquet.py` (end of `migrate_shards()`, after `print("All verifications passed!")`)
- Modify: `src/speech_spoof_bench/data/dataset_skeleton/build_parquet.py` (add a documented example call)

- [ ] **Step 1: DF builder — emit after verification**

In `benchmarks/ASVspoof2021_DF/build_parquet.py`, in `build()`, the tail currently reads:

```python
    _verify(num_shards, sample_mode)
    print("All verifications passed!")
```

Change it to:

```python
    _verify(num_shards, sample_mode)
    print("All verifications passed!")

    if not sample_mode:
        from speech_spoof_bench import labels
        out = labels.emit_labels(REPO_ROOT)
        print(f"Wrote {out}")
```

- [ ] **Step 2: 2019_LA builder — emit after verification**

In `benchmarks/ASVspoof2019_LA/build_parquet.py`, at the end of `migrate_shards()`, the tail reads:

```python
    print("All verifications passed!")
```

Change it to:

```python
    print("All verifications passed!")

    from speech_spoof_bench import labels
    out = labels.emit_labels(REPO_ROOT)
    print(f"Wrote {out}")
```

- [ ] **Step 3: Skeleton stub — document the call**

In `src/speech_spoof_bench/data/dataset_skeleton/build_parquet.py`, inside the `main()` stub body (the function that currently raises `NotImplementedError`), add this comment block immediately above the `raise`:

```python
    # After writing data/test-*.parquet, ship the fast-path labels file:
    #     from speech_spoof_bench import labels
    #     labels.emit_labels(Path(__file__).resolve().parent)
    # This lets reproduce/nightly fetch one small file instead of streaming
    # every shard. See docs/developing/new-dataset.md.
```

- [ ] **Step 4: Verify imports resolve (no build run)**

Run: `python -c "import ast; ast.parse(open('benchmarks/ASVspoof2021_DF/build_parquet.py').read()); ast.parse(open('benchmarks/ASVspoof2019_LA/build_parquet.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add benchmarks/ASVspoof2021_DF/build_parquet.py benchmarks/ASVspoof2019_LA/build_parquet.py src/speech_spoof_bench/data/dataset_skeleton/build_parquet.py
git commit -m "feat: builders emit data/labels.parquet at end of full build"
```

> `benchmarks/*` are separate dataset repos. Committing here records the change in your working tree; pushing each dataset repo to HF happens in the Manual Rollout section.

---

## Task 8: Version bump + docs

**Files:**
- Modify: `pyproject.toml:7`, `src/speech_spoof_bench/__init__.py:3`
- Modify: `docs/developing/new-dataset.md`, `docs/architecture/submission-lifecycle.md`, `docs/architecture/versioning.md`

- [ ] **Step 1: Bump the version**

In `pyproject.toml` change `version = "0.1.1"` to `version = "0.2.0"`.
In `src/speech_spoof_bench/__init__.py` change `__version__ = "0.1.1"` to `__version__ = "0.2.0"`.

- [ ] **Step 2: Verify the package reports the new version**

Run: `python -c "import speech_spoof_bench as s; print(s.__version__)"`
Expected: `0.2.0`

- [ ] **Step 3: Document the labels file + emit-labels**

In `docs/developing/new-dataset.md`, after the "Step 3 — Build the parquet" section, add:

```markdown
### `data/labels.parquet` (fast reproduction)

Ship a tiny `data/labels.parquet` (`utterance_id: string`, `label: int8`)
alongside the shards. Reproduction and nightly revalidation read this one file
instead of streaming every shard (80 HTTP round-trips → 1 for ASVspoof2021_DF).

`build_parquet.py` emits it automatically at the end of a full build. For a
dataset already built and pushed, backfill it without re-encoding audio:

    speech-spoof-bench emit-labels ./mydataset
    # reads data/test-*.parquet (notes,label only) → writes data/labels.parquet

The shards stay the source of truth: `emit-labels` asserts the file matches the
shards before writing. `reproduce` falls back to streaming shards when the file
is absent (older datasets), and `reproduce --force-shards` bypasses it.
```

In `docs/architecture/submission-lifecycle.md`, add a short paragraph under the reproduction/nightly description:

```markdown
**Label fetch fast path.** `reproduce --scoring` resolves labels in this order:
in-process cache → `data/labels.parquet` (one request) → `data/test-*.parquet`
shard stream (fallback). Labels at a pinned `dataset.revision` are immutable, so
they are memoized per `(dataset_id, revision)` for the life of the process.

**Nightly skip-unchanged.** `nightly-revalidate` skips a submission entirely when
`(scores_sha256, dataset.revision, installed bench_version)` is unchanged since
the last green run (state cached via GitHub Actions). A package release bumps
`bench_version` and re-verifies everything; a weekly `--full` sweep and the
`--full` flag force a complete pass. Trade-off: artifact drift at `scores_url`
on an otherwise-unchanged submission is only detected on the next full sweep.
```

In `docs/architecture/versioning.md`, add under the package-version section:

```markdown
- `0.2.0` — adds the `data/labels.parquet` fast path, in-process label cache,
  and nightly skip-unchanged. Additive (no schema change). Bumping the package
  version invalidates the nightly green store, forcing a full re-verify.
```

- [ ] **Step 4: Run the full test suite**

Run: `pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/speech_spoof_bench/__init__.py docs/
git commit -m "chore: bump to 0.2.0; document labels fast path + nightly skip"
```

---

## Manual Rollout (cross-repo / operational — not code in this repo)

These steps run after the package change is merged and released. They touch HF dataset repos and the `arena-manifest` / `arena` repos, and require credentials, so they are done by hand and verified between slices (matches the staged-delivery preference).

- [ ] **Release the package** and note the new commit SHA.
- [ ] **Bump the Arena pin:** update `speech-spoof-bench @ <sha>` in `arena/requirements.txt` to the released SHA and redeploy the Space (else the Space runs stale code).
- [ ] **Slice A — ASVspoof2021_DF:**
  - [ ] `speech-spoof-bench emit-labels ./benchmarks/ASVspoof2021_DF` (seconds; no re-encode).
  - [ ] Commit `data/labels.parquet` to the DF dataset repo and push to `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF`; note the new commit SHA.
  - [ ] Re-pin DF's `revision` in `arena-manifest` (+ `dataset_repin` changelog note).
  - [ ] Verify: a remote `speech-spoof-bench reproduce <a DF submission>.yaml --scoring --no-local` completes in seconds (one request), and `--force-shards` still works.
- [ ] **Slice B — ASVspoof2019_LA:** repeat the emit/commit/push/re-pin/verify steps for `ASVspoof2019_LA`.
- [ ] **Confirm** a freshly scaffolded dataset's `build_parquet.py` ships `labels.parquet` by default (Task 7 Step 3 documents the call; new builders that follow the skeleton inherit it).

---

## Self-Review

- **Spec coverage:** Layer 1 (artifact + read fast path + `emit-labels` backfill + `force_shards`) → Tasks 1–3, 7. Layer 2 (in-memory cache, no disk) → Task 2 (`_LABEL_CACHE`). Layer 3 (skip-unchanged + green store in Actions cache + `--full` + weekly sweep + logging) → Tasks 4–6. Versioning/docs/rollout → Task 8 + Manual Rollout. All spec sections map to a task.
- **Placeholder scan:** none — every code/test/command step is concrete.
- **Type consistency:** `emit_labels(dir)->Path`, `load_labels_file(path)->dict[str,int]`, `LABELS_FILENAME` used by Task 2; `_download_labels_file`, `_stream_labels_from_shards`, `_LABEL_CACHE`, and `run_scoring(..., force_shards=...)` consistent across Tasks 2–3; `green_store.{load,save,is_green,record_green,submission_id,DEFAULT_STORE_PATH,_BENCH_VERSION}` consistent across Tasks 4–5; `collect_failures(*, full=False)`, `_check_submission_data(dataset_id, data)`, `run(*, open_issues, full=False)` consistent across Task 5 + CLI.
- **Dropped-guarantee handling:** the skip's lost artifact-drift detection is mitigated by `--full`, the weekly `0 5 * * 0` sweep, and `_BENCH_VERSION` invalidation — all present in Tasks 5–6 and documented in Task 8.
