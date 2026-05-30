# Fast Label Verification — Design (tasks #10 + #11)

**Date:** 2026-05-30
**Status:** Draft for review
**Scope:** Package change to `speech-spoof-bench` (+ a per-dataset build step). Needs a
package version bump and an Arena pin bump (the code runs in CI `verify-pr`/`nightly`).

## Problem

Reproduction and nightly revalidation are **label-only** — they recompute EER from a
submission's `scores.txt` against the dataset's labels, never touching audio. But fetching
those labels is slow and repeated:

- `reproduce._stream_labels()` calls `load_dataset(..., streaming=True,
  columns=["notes","label"])`. Column projection avoids transferring audio bytes, but
  `datasets` still issues **per-shard range requests**. For ASVspoof2021_DF that's **80
  shards → ~80+ HTTP round-trips** — a remote (`--no-local`) reproduce took **~50 min** vs
  **seconds** off a local copy. Latency, not bytes, dominates.
- `nightly-revalidate` walks **every merged submission across every dataset, daily**, and
  re-streams the same immutable labels each time.
- Labels at a pinned `dataset.revision` are **immutable**, so all of this re-fetches a
  constant.

Two complementary optimizations:
- **#10 — Committed labels file:** ship a tiny labels artifact in each dataset repo → one
  small download instead of an 80-shard stream.
- **#11 — Label cache + skip-unchanged:** memoize labels per `(dataset_id, revision)` and
  skip re-verification when nothing affecting the result changed.

## Background (current code)

- `reproduce.py::_stream_labels(dataset_id, split, revision, *, force_remote)` → returns
  `{utterance_id: int_label}`. Local-registry path reads `data/test-*.parquet` with
  `columns=["notes","label"]`; remote path streams from HF at the pinned revision.
- The dataset loader globs `data/test-*.parquet` for shards, so a `labels.parquet` (no
  `test-` prefix) is NOT mistaken for a data shard — good.
- Build: `dataset-builders/<name>/build_parquet.py` (+ copy in dataset repo) emits the
  shards; the labels file is emitted here too.

## #10 — Committed per-dataset labels file

### Format: `data/labels.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `utterance_id` | `string` | scoring join key (from `notes.utterance_id`) |
| `label` | `int8` | 0=bonafide, 1=spoof (matches `ClassLabel` order) |

Chosen over `labels.txt`: typed, ~half the size, one columnar read, and `pyarrow` is
already a dep. DF: 611,829 rows ≈ 3–7 MB, one request. (`labels.txt` rejected for size
~13 MB; revisit only if a non-parquet consumer needs it.)

### Build side

After writing shards, derive from the same in-memory rows (no re-read):

```python
import pyarrow as pa, pyarrow.parquet as pq
uids   = [r["uid"] for r in records]
labels = [0 if r["label"] == "bonafide" else 1 for r in records]
pq.write_table(
    pa.table({"utterance_id": pa.array(uids, pa.string()),
              "label":        pa.array(labels, pa.int8())}),
    str(PARQUET_DIR / "labels.parquet"),
)
```

Post-write assert: row count == EXPECTED_ROWS, unique `utterance_id`, and its
`{uid: label}` map **equals** the one derived from the shards (shards stay the source of
truth; the labels file is a verified cache of them).

### Read side (`reproduce._stream_labels`)

Resolution order, **shards stay authoritative**, labels file is pure optimization:

1. **Local path:** if `<mapped>/data/labels.parquet` exists → read it; else stream shards
   (today's behavior).
2. **Remote path:** try `hf_hub_download(dataset_id, "data/labels.parquet",
   repo_type="dataset", revision=revision)`; on success read it (one request); on 404 /
   `EntryNotFoundError` (older datasets) → fall back to streaming shards. **Never fail
   because the file is absent.**

Return type unchanged (`{utterance_id: int}`); `run_scoring()` needs no change.

### Trust

The file lives inside the dataset repo at the pinned revision → same immutability guarantee
as the shards. Build-time consistency assert guarantees it matches the shards at publish.
Add a `force_shards` escape hatch on `reproduce` to bypass the file when a maintainer wants
to verify against shards directly. (Optional later: a non-fatal D8 in `validate-dataset`
warning if `labels.parquet` is inconsistent with shards.)

### Migrating existing datasets

2019_LA / 2021_DF predate the file → regenerate + re-pin in the manifest (new revision SHA
+ `dataset_repin` changelog note). Old submissions stay reproducible against the old
revision (streams shards); new runs use the fast path. Opt-in per dataset.

## #11 — Label cache + skip-unchanged re-verification

### Label cache (per `(dataset_id, revision)`)

Immutable labels → memoize. Key `(dataset_id, revision)`; store `{uid: label}` under
`~/.cache/speech_spoof_bench/labels/<org>__<name>@<rev>.parquet`. `_stream_labels()` checks
cache first; miss → fetch (via #10 file or shard stream) → write cache. Add a process-level
dict so repeated calls in one `nightly` run skip disk. New revision = new key (auto
invalidation).

### Skip-unchanged (`nightly`)

Key each submission's verification by `(scores_sha256, dataset.revision, bench_version)`.
If unchanged since the last **green** verification, skip recompute (result can't have
moved). Store a "last verified" record (committed back, or CI cache). A `bench_version`
bump forces full re-verify (drift detector working). `log()` skipped vs re-verified so a
silent skip never hides a problem. Turns nightly from O(submissions × fetch) into
O(changed submissions).

## Files to change

| File | Change |
|------|--------|
| `dataset-builders/<name>/build_parquet.py` + dataset-repo copies | emit `data/labels.parquet` + consistency assert |
| `src/speech_spoof_bench/reproduce.py` | `_stream_labels`: labels-file fast path + cache; `force_shards` escape hatch |
| `src/speech_spoof_bench/ci/nightly.py` | skip-unchanged via `(scores_sha256, revision, bench_version)` |
| `pyproject.toml` + `__init__.py` | minor version bump |
| `arena/requirements.txt` | bump pin to new package SHA after release |
| `docs/architecture/{submission-lifecycle,versioning}.md` | document labels file + cache |
| tests | labels round-trip, fallback-when-absent, cache hit/miss, nightly skip |

## Versioning impact

- **Package:** minor bump (`0.1.1 → 0.2.0`) — additive feature.
- **Arena pin:** bump required (ingest/nightly run in CI/Space) → update
  `arena/requirements.txt`, redeploy Space.
- **Dataset revision:** re-pin per dataset when `labels.parquet` added (+ changelog).
- No submission/result/manifest **schema** change (labels file is data).

## Rollout order

1. Land package change (read-side fallback + cache + nightly skip) — safe with/without the
   file (graceful fallback). Release + bump Arena pin.
2. Regenerate ASVspoof2021_DF with `labels.parquet`, re-pin; verify reproduce uses fast
   path (one request, seconds).
3. Same for ASVspoof2019_LA.
4. Add `labels.parquet` to the dataset scaffold so new datasets ship it by default.

## Out of scope (YAGNI)

- Streaming `scores.txt` parser (only matters at 100M+ rows).
- Replacing EER recompute (trivial vs I/O).
- Signed/checksummed labels file beyond the build-time assert.
