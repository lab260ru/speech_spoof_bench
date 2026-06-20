# Arena dataset plan: CFAD

# Plan (reviewed at the 🚦 PLAN GATE — before any probe/build/push)

- **Dataset Name:** `CFAD`  (source casing → `benchmarks/CFAD/`, HF repo
  `SpeechAntiSpoofingBenchmarks/CFAD`, manifest id `SpeechAntiSpoofingBenchmarks/CFAD`; no lowercase slug)
- **Raw source (read-only):** `/home/kirill/mnt/users_4tb/datasets/CFAD/CFAD/clean_version/`
  — partitions `test_seen_clean/` and `test_unseen_clean/`. **No protocol file** — label is encoded
  in the directory tree (`fake_clean/` = spoof, `real_clean/` = bonafide).
- **Scope:** the **clean** test sets, both seen + unseen, as requested. The `codec_version/` and
  `noisy_version/` robustness variants and the train/dev splits are **excluded** (could be a future
  Extended add).
- **Date:** 2026-06-07

## Protocol → schema mapping
- **id** → `utterance_id` = `CFAD_` + relpath-under-`clean_version` with `/`→`_`, no `.wav`
  (e.g. `CFAD_test_seen_clean_fake_clean_gl_SSB07800001_gl`). Path-based ⇒ guaranteed unique
  (filename stems repeat across method dirs; the full relpath is the immutable join key).
- **`path`** column = the relpath under `clean_version` (unique).
- **label** → directory: anything under `*/fake_clean/**` = **spoof**, under `*/real_clean/**` = **bonafide**.
- **`notes`** JSON fields to keep: `utterance_id`, `partition` (`test_seen`|`test_unseen`),
  `source` (the method/corpus dir, e.g. `gl`, `hifigan`, `aishell1`, `world`, `partiallyfake`).
- **Expected counts (after dropping the 1 outlier):** total **62,999** / bonafide **20,999** / spoof **42,000**
  - on-disk: 63,000 = 42,000 spoof + 21,000 bonafide; minus the dropped 97-min aishell1 clip → 20,999 bonafide

## ⚠️ One data anomaly — DECIDED AT GATE: DROP
- `test_seen_clean/real_clean/aishell1/BAC009S0764W0123.wav` is **178 MB ≈ 97 minutes** (a known
  aishell1 concatenation quirk). The *only* file >5 MB; no empty files.
- **Gate decision: DROP it.** New expected counts: total **62,999** / bonafide **20,999** / spoof **42,000**.

## License & redistribution
- **CC BY 4.0** → SPDX/HF tag `cc-by-4.0`; redistribution **permitted** with attribution. Ship the
  verbatim CC BY 4.0 `LICENSE.txt` (reuse `benchmarks/CD-ADD/LICENSE.txt`, already cc-by-4.0).
- Paper / arXiv: **2207.12308** (real arXiv id → goes in the `arxiv` front-matter key; no DOI fallback).

## Manifest placement
- **Set: Core (default)** — re-computes coverage for every existing submission (flagged for maintainer).

## Build approach (clean vs re-encode)
- Source: **16 kHz mono PCM_16 WAV** (already canonical SR), clip durations ~2–5 s (+ the one outlier).
- Path: **GATE DECISION — re-encode to FLAC** (fork `benchmarks/SONAR/build_parquet.py`: walk the
  tree, two-stage librosa→16 kHz mono FLAC staging + shard assembly; whole-set decode probe is folded
  into Stage-1 re-encode per [[reference_dataset_reencode_sr_and_probe]]). Non-silence guard on.
- Parquet ≈ **~4–5 GB** FLAC.
- Shard sizing: ~350 MB/shard → **~14 shards**. Read **sorted by `utterance_id`** (HDD sequential),
  ~64 workers; resumable + atomic per shard; delete stale `-of-NNNNN` shards; assert counts;
  emit `data/labels.parquet`.

## 🚦 PLAN GATE — present the above; await explicit OK. Probe/build/push nothing before this.

---

# Execution log (filled autonomously after approval)

- [x] Scaffolded at `benchmarks/CFAD/`; LICENSE (cc-by-4.0 from CD-ADD)/.gitattributes/.gitignore copied
- [x] **Decode probe folded into Stage-1 re-encode:** 0/62999 failures → re-encode FLAC
- [x] Read params: 88 workers, sorted by uid; Stage-1 ~5.5 min; **14 shards**; **3.9 GB**; counts asserted (20999/42000/62999)
- [x] `validate-dataset ./CFAD --skip-submissions` → D1–D7 green (62999 ids/paths unique)
- [x] Pushed `SpeechAntiSpoofingBenchmarks/CFAD` @ `53d7855c1c378524f7b7b1030bcb6b2caa327fe6`; online sanity (stream rows @16k + labels.parquet 62999 rows) OK
- [x] Manifest PR (`core_set` @ `53d7855` + CHANGELOG `dataset_added`): https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/arena-manifest/discussions/6 — built from fresh main (no local clone edited; LibriSeVoc/SONAR preserved)
- [x] Random-baseline seeded (via `submitting-arena-model`): submission PR https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/CFAD/discussions/1; **EER 49.91%** (62,999 trials, n_skipped=0; reproduce self-consistency 49.906%≈49.907%). scores @ random-baseline-asas commit `0e99733`. verify/merge waits on maintainer ingest (manifest #6 unmerged).

## Maintainer to-dos (surface in final report)
- Review/merge the manifest PR (Core changes everyone's coverage)
- Re-ingest to subscribe the webhook
- Merge the random-baseline submission + fill its reproduction block (its verify-pr routes only after re-ingest)

## Notes / guideline discrepancies
- submit-dataset.md says new datasets "normally start in Extended"; the skill overrides to **Core by
  default**. Following the skill (Core).
