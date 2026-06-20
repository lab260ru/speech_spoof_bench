# Arena dataset plan: DeepVoice

---

# Plan (reviewed at the 🚦 PLAN GATE — before any probe/build/push)

- **Dataset Name:** `DeepVoice`  (source casing — used verbatim for `benchmarks/DeepVoice/`, HF repo `SpeechAntiSpoofingBenchmarks/DeepVoice`, and the manifest id; no lowercase slug)
- **Raw source (read-only):** audio dir `/home/kirill/mnt/users_4tb/datasets/deep-voice/KAGGLE/AUDIO/{FAKE,REAL}`; **no protocol file** — label is the top-level directory (`REAL/` = bonafide, `FAKE/` = spoof). The `KAGGLE/DATASET-balanced.csv` is a precomputed MFCC/feature table, **not** a file index — ignored; we walk the tree.
- **Date:** 2026-06-09
- **Paper:** DEEP-VOICE (arXiv 2308.12734). FAKE clips are Retrieval-based Voice Conversion (RVC) speaker-to-speaker; REAL are the 8 source speakers' originals.

## Protocol → schema mapping
- id column → `utterance_id`: synthesized as `DEEPVOICE_{REAL|FAKE}_{filename-stem}` (filenames are unique within each dir and there are **0 cross-dir collisions**; the label-prefix makes ids robust + self-describing). Stem e.g. `Obama-to-Biden_frag0`, `biden-original_frag12`.
- label column → bonafide/spoof: directory — `REAL/` → bonafide, `FAKE/` → spoof.
- extra `notes` fields to keep: `source_rel_path`, `split` (`REAL`/`FAKE`), `source_stem` (e.g. `biden-to-linus`, `obama-original`) for provenance.
- **Expected counts:** total **5053** / bonafide **628** (REAL) / spoof **4425** (FAKE)

## License & redistribution
- SPDX / HF tag: `mit` (per user). MIT permits redistribution of the audio (verbatim `LICENSE.txt` shipped). redistribution permitted: **yes — MIT**.

## Manifest placement
- **Set:** Core (default) — Core re-computes coverage for every existing submission (expected; flagged in PR).

## Build approach (confirm clean vs re-encode after the whole-set probe)
- Source SR / format: **stereo, 40000 Hz & 48000 Hz, 16-bit PCM WAV** (sampled `file` on both dirs' row-0). This is non-16 kHz and 2-channel → a raw-byte embed would ship bytes that fail the validator's D3 / row-0 16 kHz check.
- Path: **re-encode** (librosa → clean 16 kHz mono FLAC staging, then assemble shards) — forking the SONAR re-encode builder. Whole-set soundfile decode probe runs as a reporting gate (folded into Stage-1 re-encode); non-silence guard rejects all-zero re-encodes. (Finalized after the whole-set probe, but stereo/40k/48k already forces re-encode.)
- Shard sizing: ~5 GB stereo source → ~1–1.5 GB of 16 kHz mono FLAC → **~3–4 shards** at ~350 MB each (finalize `NUM_SHARDS` from actual staging size during build). Reference dir to fork: `benchmarks/SONAR/`.

## 🚦 PLAN GATE — present the above; await explicit OK. Probe/build/push nothing before this.

---

# Execution log (filled autonomously after approval)

- [x] Scaffolded at `benchmarks/DeepVoice/`; LICENSE (MIT)/.gitattributes/.gitignore copied
- [x] **Whole-set decodability probe:** 0/5053 failed → path = re-encode (source confirmed stereo 40/48 kHz PCM)
- [x] Read params: 64 workers, uid-sorted order; Stage1 re-encode 0 failures; 2 shards (268 MB + 257 MB); counts asserted 5053/628/4425
- [x] `validate-dataset ./DeepVoice --skip-submissions` → D1–D7 green (5053 ids, 5053 paths)
- [x] Pushed `SpeechAntiSpoofingBenchmarks/DeepVoice` @ `cc3bdf544cfc09bd9cc788f7f022ba1af9daf701`; online sanity (stream rows 16 kHz decodable + labels.parquet 5053/628/4425) OK
- [x] Manifest PR (`core_set` @ sha + CHANGELOG `dataset_added`): https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/arena-manifest/discussions/15 ; edited fresh main copies (local clone was stale — missing DECRO — never touched it)
- [x] Random-baseline seeded (via `submitting-arena-model`): submission PR https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/DeepVoice/discussions/1 ; EER 48.73% (5053 trials, 0 skipped); reproduce self-consistency exact match; scores pinned at random-baseline-asas@feb23ccb105938f93e40247da4799e2b2c76aacc

## Maintainer to-dos — DONE (executed by request)
- [x] Merged manifest PR #15 — live core_set = 13 datasets, DeepVoice present, DECRO preserved (no silent drop; main was still at #14 DECRO, no rebase needed)
- [x] Verified submission via `reproduce --scoring` (sha matched, EER Δ 0.0, 5053 trials / 0 skipped); filled reproduction block (`match: scoring`) on PR #1 branch; merged PR #1
- [x] Re-ingested: rebuilt ArenaState (144 rows, 0 warnings, DeepVoice row present), committed cache.json to the Space (oid ba5fd55b). Space cache now has 13 datasets incl. DeepVoice @ EER 48.73%

## Notes / guideline discrepancies
- arena-manifest local clone was stale (behind main by the DECRO add). Fetched fresh `manifest.yaml`/`CHANGELOG.yaml` from remote `main` before editing, per the manifest-stale-clone-overwrite gotcha — avoided dropping DECRO. Local clone left untouched (PR holds the change).
- Submission YAML `submitted_at` must be quoted (`"2026-06-09"`) to pass strict jsonschema (bare YAML date → `date` object). SONAR's unquoted value works only because the package loader coerces; quoting is strictly safer.
