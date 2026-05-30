# Fast Label Verification — Design (tasks #10 + #11)

**Date:** 2026-05-30
**Status:** Approved for planning
**Scope:** Package change to `speech-spoof-bench` (+ a per-dataset backfill step). Needs a
package version bump and an Arena pin bump (the code runs in CI `verify-pr`/`nightly`).

## Problem

Reproduction and nightly revalidation are **label-only** — they recompute EER from a
submission's `scores.txt` against the dataset's labels, never touching audio. But fetching
those labels is slow and repeated:

- `reproduce._stream_labels()` calls `load_dataset(..., streaming=True,
  columns=["notes","label"])`. Column projection avoids transferring audio bytes, but
  `datasets` still issues **per-shard range requests**. For ASVspoof2021_DF that's **80
  shards → ~80+ HTTP round-trips** — a remote (`--no-local`) reproduce took **~50 min** vs
  **seconds** off a local copy. Latency, not bytes, dominates. (2019_LA: 9 shards.)
- `nightly-revalidate` walks **every merged submission across every dataset, daily**, and
  re-streams the same immutable labels each time, then re-runs an EER that cannot have
  moved unless the metric code changed.
- Labels at a pinned `dataset.revision` are **immutable**, so all of this re-fetches a
  constant.

Three complementary optimizations, all approved for this spec:

- **Layer 1 (#10) — Committed labels file:** ship a tiny `labels.parquet` in each dataset
  repo → one small download instead of an 80-shard stream.
- **Layer 2 (#11a) — In-memory label cache:** memoize labels per `(dataset_id, revision)`
  in a process-level dict so repeated calls in one run skip re-fetching. **No disk cache.**
- **Layer 3 (#11b) — Skip-unchanged nightly:** skip a submission's verification entirely
  when `(scores_sha256, revision, bench_version)` is unchanged since the last green run.

## Decisions locked during brainstorming

1. **Labels artifact format:** `data/labels.parquet` (typed, ~3–7 MB for DF, one columnar
   read, `pyarrow` already a dep) — not a `.txt`. Both formats collapse 80 round-trips to
   one request; parquet is smaller and typed.
2. **Generation path:** a **standalone `emit-labels` command** that derives the file from
   existing shards (no audio re-encode). `build_parquet.py` calls the same function at the
   end of a full build so new datasets ship it automatically.
3. **Scope:** all three layers.
4. **Nightly skip semantics:** **skip everything, no network** when unchanged (fastest;
   O(changed submissions)). Explicitly trades away re-detecting `scores_url` artifact drift
   on unchanged submissions — mitigated by `--full`, a `bench_version` bump, and a
   scheduled weekly full sweep.
5. **Green store location:** **GitHub Actions cache** (`actions/cache`), not a committed
   file. The skip is a pure optimization, so cache loss only triggers a safe re-verify.

## Background (current code)

- `reproduce.py::_stream_labels(dataset_id, split, revision, *, force_remote)` → returns
  `{utterance_id: int_label}`. Local-registry path (`local_registry.lookup` → `Path`) reads
  `data/test-*.parquet` with `columns=["notes","label"]`; remote path streams from HF at
  the pinned revision.
- The dataset loader globs `data/test-*.parquet` for shards, so a `labels.parquet` (no
  `test-` prefix) is NOT mistaken for a data shard — good.
- `nightly.py::collect_failures()` lists submissions per manifest dataset and calls
  `_check_submission` → `reproduce.run_scoring(...)` for each.
- Version lives in `pyproject.toml` (`0.1.1`) and `src/speech_spoof_bench/__init__.py`
  (`__version__`).
- CLI parsers live in `cli.py`: `reproduce` (rp), `ci nightly-revalidate` (nr),
  `scaffold-dataset`, `validate-dataset`, etc.

## Layer 1 — Committed per-dataset labels file

### Format: `data/labels.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `utterance_id` | `string` | scoring join key (from `notes.utterance_id`) |
| `label` | `int8` | 0=bonafide, 1=spoof (matches `ClassLabel` order) |

### `emit-labels` — new module + command

New `src/speech_spoof_bench/labels.py` with `emit_labels(dataset_dir: Path) -> Path`:

1. Glob `<dir>/data/test-*.parquet`; read each with `columns=["notes","label"]` only (no
   audio bytes transferred).
2. Build `{utterance_id: int_label}` from `notes.utterance_id` + `label`.
3. Write `<dir>/data/labels.parquet` (`utterance_id: string`, `label: int8`):

```python
import pyarrow as pa, pyarrow.parquet as pq
pq.write_table(
    pa.table({"utterance_id": pa.array(uids, pa.string()),
              "label":        pa.array(labels, pa.int8())}),
    str(data_dir / "labels.parquet"),
)
```

4. **Consistency asserts** (shards stay the source of truth; the file is a verified cache):
   row count == sum of shard row counts, `utterance_id` values unique, and the
   `{uid:label}` map equals the one re-derived from the shards. Fail loudly otherwise.

`build_parquet.py` (both dataset repos + the package scaffold copy) calls
`labels.emit_labels(REPO_ROOT)` at the end of a full build so new datasets ship the file by
default. Existing datasets get it via a seconds-long `emit-labels` backfill — no re-encode.

### Read side — `reproduce._stream_labels`

Resolution order; shards stay authoritative, labels file is pure optimization, **never
fails because the file is absent**:

1. **Process cache (Layer 2) hit** → return immediately.
2. **Local path:** if `<mapped>/data/labels.parquet` exists → read it; else stream shards
   (today's behavior).
3. **Remote path:** try `hf_hub_download(dataset_id, "data/labels.parquet",
   repo_type="dataset", revision=revision)` → read it (one request); on 404 /
   `EntryNotFoundError` (older datasets) → fall back to streaming shards.
4. Populate the process cache before returning.

Return type unchanged (`{utterance_id: int}`); `run_scoring()` needs no change.

### `force_shards` escape hatch

`run_scoring(..., force_shards: bool = False)` and a `reproduce --force-shards` CLI flag
bypass `labels.parquet` (and the cache) to verify directly against shards when a maintainer
wants to.

### Trust

The file lives inside the dataset repo at the pinned revision → same immutability guarantee
as the shards. The build/backfill consistency assert guarantees it matches the shards at
publish. (Optional later: a non-fatal warning in `validate-dataset` if `labels.parquet` is
inconsistent with shards.)

## Layer 2 — In-memory label cache (no disk)

Immutable labels → memoize per `(dataset_id, revision)` in a **process-level dict** in
`reproduce`:

- First call fetches (via the Layer-1 file or a shard stream) and stores `{uid:label}`.
- Subsequent calls in the same process return instantly.
- New revision = new key → automatic invalidation.
- **No `~/.cache` file** — nothing to invalidate, clean up, or get stale on disk. A fresh
  process starts empty; within one `nightly` run the dict still collapses N
  submissions-per-dataset into one label fetch, which is the case that mattered.

Resolution order: **process dict → `labels.parquet` (one request) → shard stream**.

## Layer 3 — Skip-unchanged nightly

### Green store (GitHub Actions cache)

`nightly-green.json`, persisted via `actions/cache` in `nightly-revalidate.yml`:

```json
{ "<dataset_id>/<slug>": {"scores_sha256": "...", "revision": "...",
                          "bench_version": "0.2.0", "ts": "..."} }
```

Cache key includes `bench_version`. Loss (7-day inactivity / 10 GB eviction) only triggers
a safe re-verify that repopulates it.

### Skip logic (`nightly.collect_failures`)

For each submission, read the YAML (already done; gives `scores_sha256` + `revision` with
no extra network):

- If `(scores_sha256, revision, bench_version)` matches a green-store entry → **skip
  entirely**: no `scores_url` download, no label fetch, no recompute.
- Else → full `run_scoring`; on green, write/refresh the store entry.
- `bench_version` = `speech_spoof_bench.__version__`. Any package release re-keys every
  entry → full re-verify (the drift detector for code/label changes still works).
- `log()` skipped-vs-reverified counts so a silent skip never hides a problem.

Turns nightly from O(submissions × fetch) into O(changed submissions).

### Dropped guarantee + mitigations

"Skip everything" means unchanged submissions are no longer re-fetched from `scores_url`,
so **artifact drift / dead links on unchanged submissions go undetected** until something
changes. Mitigations:

- `ci nightly-revalidate --full` flag → forces a complete sweep, ignoring the store.
- A `bench_version` bump (any release) auto-busts the store.
- A scheduled **weekly `--full`** cron line in the workflow to catch URL drift periodically.

## Files to change

| File | Change |
|------|--------|
| `src/speech_spoof_bench/labels.py` (new) | `emit_labels(dir)` + consistency asserts |
| `src/speech_spoof_bench/reproduce.py` | `_stream_labels`: labels-file fast path + process cache; `run_scoring(force_shards=...)` |
| `src/speech_spoof_bench/ci/nightly.py` | skip-unchanged via green store; `--full` path; skipped/reverified logging |
| `src/speech_spoof_bench/cli.py` | `emit-labels` subcommand; `reproduce --force-shards`; `ci nightly-revalidate --full` |
| `.github/workflows/nightly-revalidate.yml` | `actions/cache` for `nightly-green.json`; weekly `--full` schedule |
| `dataset-builders/<name>/build_parquet.py` + dataset-repo copies | call `labels.emit_labels` at end of full build |
| `pyproject.toml` + `__init__.py` | minor version bump `0.1.1 → 0.2.0` |
| `arena/requirements.txt` | bump pin to new package SHA after release |
| `docs/architecture/{submission-lifecycle,versioning}.md`, `docs/developing/new-dataset.md` | document labels file, `emit-labels`, cache, nightly skip |
| tests | labels round-trip + assert failures; fallback-when-absent; `force_shards`; process-cache hit; nightly skip / re-verify / `--full` |

## Versioning impact

- **Package:** minor bump (`0.1.1 → 0.2.0`) — additive feature.
- **Arena pin:** bump required (ingest/nightly run in CI/Space) → update
  `arena/requirements.txt`, redeploy Space.
- **Dataset revision:** re-pin per dataset when `labels.parquet` is added (+ changelog
  `dataset_repin` note).
- No submission/result/manifest **schema** change (labels file is data).

## Rollout order (staged, verify between slices)

1. Land package change (read-side fallback + process cache + nightly skip + CLI flags) —
   safe with or without the file (graceful fallback). Release `0.2.0`, bump Arena pin.
2. `emit-labels ./benchmarks/ASVspoof2021_DF` → commit `labels.parquet` → push → re-pin in
   manifest → verify a remote `reproduce` is now one request / seconds.
3. Same for `ASVspoof2019_LA`.
4. Confirm `build_parquet.py` calls `emit_labels` so new datasets ship it by default.

## Out of scope (YAGNI)

- Disk-persistent label cache (in-memory only this round).
- Committed green store / write-back machinery (using `actions/cache`).
- Streaming `scores.txt` parser (only matters at 100M+ rows).
- Replacing EER recompute (trivial vs I/O).
- Signed/checksummed labels file beyond the build-time assert.
