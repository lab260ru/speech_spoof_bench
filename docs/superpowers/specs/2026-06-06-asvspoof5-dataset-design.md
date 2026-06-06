# ASVspoof5 Benchmark — Design

**Date:** 2026-06-06
**Goal:** Package the ASVspoof5 eval split (track_1) as an Arena-ready benchmark under
`benchmarks/ASVspoof5/`, validate it all-green, push it to the HF org, and add it to the
manifest's **Core** set. After successful completion, wrap the repeatable procedure into a
skill (separate follow-up task — see "Skill follow-up").

## Source

- Root: `/home/kirill/mnt/users_4tb/datasets/asvspoof5/last_eval` — **read-only; never modified.**
- `flac_E_eval/` — 681,872 source FLAC files (≈1k extra beyond the protocol; we build only
  from the protocol rows).
- `ASVspoof5.eval.track_1.tsv` — **680,774** rows, 10 space-separated columns:
  `speaker_id  utterance_id  gender  codec  col5  source_id  AC1  attack_id  label  col10`.
  Labels: **138,688 bonafide / 542,086 spoof**.
- `track_2.*` (SASV enroll/trial) — **out of scope** (different task, not bonafide/spoof binary).

**Probe finding (decisive):** source FLAC is already **16 kHz mono**, clips **4–10 s**,
**0/30 soundfile decode failures** → take the **CLEAN path: embed raw source bytes
directly** (no decode/re-encode). This is the fast branch; 2021_LA's re-encode stage is skipped.

## Decisions (locked with user)

| Decision | Choice |
|----------|--------|
| Scope | **Full** track_1 eval set (all 680,774) |
| Name | `ASVspoof5` |
| Deliverable | Build + validate offline → push to HF → open arena-manifest PR |
| License | **ODC-By v1.0** (same `LICENSE.txt` as existing ASVspoof repos) |
| Manifest set | **Core** (deliberately re-computes every submission's coverage) |

## Target artifact

Mirror the canonical dataset-repo shape (see `docs/developing/new-dataset.md`), using
`benchmarks/ASVspoof2021_LA/` as the reference implementation:

```
benchmarks/ASVspoof5/
├── README.md            # ODC-By front-matter, arena-ready tag, stats, provenance, citation
├── eval.yaml            # task=antispoofing_eval, metrics=[eer_percent]
├── LICENSE.txt          # ODC-By v1.0 (copied from 2021_LA)
├── build_parquet.py     # forked + trimmed from 2021_LA (clean path only effectively)
└── data/
    ├── test-*.parquet   # 4-column canonical schema, ~300 MB/shard
    └── labels.parquet   # emitted at end of full build
```

Canonical 4 columns:

| Column | Value |
|--------|-------|
| `path` | `<utterance_id>.flac` (col 2) — unique |
| `audio` | raw source FLAC bytes via `Audio(sampling_rate=16000)` (already 16 kHz, no resample) |
| `label` | `ClassLabel(["bonafide","spoof"])` from col 9 |
| `notes` | JSON: `utterance_id` (unique join key), `speaker_id`, `gender`, `codec`, `attack_id`, and remaining protocol fields preserved |

## Builder design (`build_parquet.py`)

Forked from `benchmarks/ASVspoof2021_LA/build_parquet.py`, simplified:

- `SRC_ROOT = /home/kirill/mnt/users_4tb/datasets/asvspoof5/last_eval`,
  `FLAC_DIR = SRC_ROOT/flac_E_eval`, `META_PATH = SRC_ROOT/ASVspoof5.eval.track_1.tsv`.
- `parse_metadata()`: split each line on whitespace, expect 10 fields, map columns above.
- Assertions: `EXPECTED_ROWS=680774`, `EXPECTED_BONAFIDE=138688`, `EXPECTED_SPOOF=542086`.
- Sort by `utterance_id`; `_ensure_long_first_row` (all clips ≥1 s, so a no-op safeguard).
- `probe_decodability()` retained: confirms CLEAN, then `_audio_bytes` reads raw source bytes.
  The re-encode (Stage 1) code is retained but will not run on this clean source.
- Stage 2: one worker per shard, atomic `os.replace`, resumable (skip non-empty shards).
- `--limit N` / env sample mode → single shard, skip count asserts (fast offline validate).
- At end of full build: `speech_spoof_bench.labels.emit_labels()` → `data/labels.parquet`.
- **Writes only under `benchmarks/ASVspoof5/`. Source is never written.**

## Fastest-params benchmarking (explicit user ask)

Before the full run, time a 2-shard slice at a few worker counts (e.g. 8 / 16 / 32 and
`os.cpu_count()`) to find the 4 TB mount's read-throughput sweet spot, then launch the full
build (in the background) with the winning `WORKERS` and a shard count targeting ~300 MB/shard
(≈40 GB total ⇒ ~130 shards; exact count fixed after a size probe). Report timing + final size.

## Validate → push → manifest

1. `speech-spoof-bench validate-dataset ./benchmarks/ASVspoof5 --skip-submissions` → all-green
   (D1–D7). Iterate on the local dir (fast, offline) before any push.
2. Push to `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof5` (LFS for parquet).
3. `validate-dataset SpeechAntiSpoofingBenchmarks/ASVspoof5` (online) → confirm green. Note SHA.
4. Open a PR on `arena-manifest` adding ASVspoof5 to **`core_set`** with the pinned SHA, plus a
   `dataset_added` `CHANGELOG.yaml` note. Revision must be lowercase hex 7–40 chars. No
   `schema_version`/`ranking_version` bump (data change). Core addition is called out in the PR.

## Risks / watch-items

- **Upload size** (~40 GB) — large HF LFS push; ensure stable connection / resumable.
- **Core coverage shift** — adding to Core changes every existing submission's coverage and can
  move the global ranking; flagged in the manifest PR.
- **License** — proceeding on user's instruction (ODC-By v1.0); if ASVspoof5's actual terms are
  non-commercial/no-derivatives this would need revisiting before the public push.
- **Disk throughput** — the 4 TB mount may be the bottleneck; benchmarking step addresses it.

## Skill follow-up (after this task succeeds)

Once the dataset is live and verified, wrap the procedure (scaffold → fork builder for a new
raw source → probe → benchmark params → build → validate → push → manifest PR) into a skill so
future ASVspoof-style datasets can be added repeatably. Designed/implemented as its own task.
