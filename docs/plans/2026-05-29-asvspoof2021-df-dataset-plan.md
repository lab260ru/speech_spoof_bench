# ASVspoof2021_DF Dataset — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish `SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF` (full 611,829-utterance
DeepFake eval set), add it to the Arena `core_set`, and land a reproduced `random-baseline`
result through the CI/CD loop.

**Architecture:** A mostly *operational* plan — build raw flac + `trial_metadata.txt` into
canonical 4-column parquet (mirroring `ASVspoof2019_LA`), push to HF, update the
`arena-manifest` repo, then drive the existing webhook → verify-pr → post-merge-badge
pipeline by opening a submission PR. Correctness is gated by `build_parquet.py`'s built-in
asserts, `tests/test_schema.py`, the `validate-dataset` D-checks, and `reproduce --scoring`
— not by per-function unit TDD (little new library code is written).

**Tech Stack:** Python (`datasets`, `pyarrow`, `soundfile`), `speech-spoof-bench` CLI,
Hugging Face Hub + LFS, GitHub Actions (existing workflows).

Spec: [2026-05-29-asvspoof2021-df-dataset-design.md](./2026-05-29-asvspoof2021-df-dataset-design.md)

---

## Overview

Build, publish, and wire up `SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF` (full DeepFake
eval set, 611,829 utterances), mirroring `ASVspoof2019_LA`. Then add it to the Arena
manifest `core_set`, run the `random-baseline` model on it, and land the result through the
CI/CD loop. Includes a docs correction for D3's first-row-only behavior.

## Prerequisites

- `speech-spoof-bench` CLI installed (confirmed: `/home/kirill/miniconda3/bin/speech-spoof-bench`).
- Source data present at `/home/kirill/mnt/users_4tb/datasets/asvspoof2021_DF/` (confirmed).
- Target drive `/home/kirill/mnt/drive3_8tb` has ~4.7 TB free (confirmed).
- HF push credentials (the org token embedded in the 2019 repo's remote works for org repos).
- `soundfile`, `datasets`, `pyarrow` available (used by the 2019 build).

## Phase 1 — Scaffold the dataset repo

**Goal:** A local dataset dir mirroring `ASVspoof2019_LA`, with everything except the
parquet shards.

### Tasks
- [ ] Create `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF/`.
- [ ] Copy + adapt from 2019: `eval.yaml` (name/description → DF), `.gitattributes`,
      `.gitignore`, `submissions/{README.md, results_template.yaml}`, `tests/test_schema.py`.
- [ ] `LICENSE.txt` ← source `ASVspoof2021_DF_eval/LICENSE.DF.txt` (ODbL).
- [ ] `protocols/ASVspoof2021.DF.cm.eval.trl.txt` ← copy from source.
- [ ] `README.md` with full front-matter (D6 keys + `arena-ready`, `arxiv: ["2109.00537"]`,
      `size_categories: [100K<n<1M]`) + body (overview, ODbL note, schema, stats, provenance,
      both citations — arXiv `@misc` and Interspeech `@inproceedings`).
- [ ] `submissions/results_template.yaml`: dataset id → DF, `n_trials: 611829`.

### Verification
- [ ] `eval.yaml` `metrics: [eer_percent]`; README front-matter has all 8 D6 keys + tag.
- [ ] Tree matches the spec's layout (minus `data/`).

## Phase 2 — Build the parquet

**Goal:** `data/test-*.parquet` with exactly 611,829 rows in the canonical 4-col schema.

### Tasks
- [ ] Write `build_parquet.py` per spec §"build_parquet.py": parse `trial_metadata.txt`,
      join flac, sort by uid, rotate a ≥1.0 s clip to row 0, rich `notes` JSON, ~80 shards.
- [ ] Run it (background — long; ~36 GB output). Capture stdout for the assert results.
- [ ] Copy `build_parquet.py` into `dataset-builders/ASVspoof2021_DF/` with a short README.

### Verification
- [ ] Script's post-write asserts pass: total rows == 611,829; unique uids/paths;
      shard-0 columns == `{path, audio, label, notes}`; shard-0 row-0 sr==16000, dur ≥ 1.0 s.
- [ ] `bonafide`/`spoof` counts == 22,617 / 589,212.

## Phase 3 — Validate locally + smoke test

**Goal:** All-green D1–D7 offline, baseline runs end-to-end.

### Tasks
- [ ] `speech-spoof-bench validate-dataset <dir> --skip-submissions` → fix until all-green.
- [ ] `speech-spoof-bench local set SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF <dir>`.
- [ ] Smoke run: `run --model-module speech_spoof_bench.examples.random_baseline:RandomBaseline
      --datasets SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF --output-dir ./results`.

### Verification
- [ ] Validator reports D1–D7 all pass.
- [ ] `results/ASVspoof2021_DF/result.yaml`: `eer_percent ≈ 50`, `n_trials = 611829`,
      `n_skipped` small (< 5%).

## Phase 4 — Publish to HF + online validate

**Goal:** Dataset live on HF, validated against what HF serves, SHA recorded.

### Tasks
- [ ] `git init` the dataset dir (if needed), LFS-track parquet, commit, add the HF dataset
      remote (org token), push to `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF`.
- [ ] Record the commit **SHA** (lowercase hex).
- [ ] `speech-spoof-bench validate-dataset SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF` (online).

### Verification
- [ ] Online validate is all-green.
- [ ] Repo browsable on HF with ~80 parquet shards via LFS.

**[MANUAL CHECKPOINT]** — confirm the repo looks right on HF before touching the manifest.

## Phase 5 — Add to the Arena manifest

**Goal:** Dataset in `core_set`; Arena re-ingests and displays it.

### Tasks
- [ ] In `arena-manifest/manifest.yaml`, add the DF entry to `core_set` with the pinned SHA.
- [ ] Add a `CHANGELOG.yaml` `dataset_added` event (date 2026-05-29).
- [ ] Commit + push `arena-manifest` (fires the webhook → Arena refresh → DF subscribed).

### Verification
- [ ] Manifest validates (revision matches `^[0-9a-f]{7,40}$`).
- [ ] Arena shows ASVspoof2021_DF on the Datasets tab after refresh (may take a few min).

**[MANUAL CHECKPOINT]** — confirm the dataset appears on the Arena.

## Phase 6 — Run model + author submission

**Goal:** A reproduced `random-baseline` result on DF, scores uploaded to the model repo.

### Tasks
- [ ] Re-run the baseline with `--no-local --no-skip-existing` against the pinned HF revision
      to get the canonical `scores.txt` + `result.yaml`.
- [ ] Upload `scores.txt` to `SpeechAntiSpoofingBenchmarks/random-baseline-asas` under
      `.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF/scores.txt`; record its commit SHA.
- [ ] Author `submissions/random-baseline.yaml` (slug `random-baseline`, pinned `scores_url`
      with that SHA, `scores_sha256`, `n_trials: 611829`, dataset `revision` = Phase-4 SHA,
      `bench_version` from result.yaml, empty `reproduction: {}`).
- [ ] `speech-spoof-bench reproduce <submission>.yaml --scoring --no-local` (mirror CI).

### Verification
- [ ] `reproduce` passes: sha matches, recomputed EER == claimed within 1e-6, coverage OK.

## Phase 7 — Open submission PR + CI/CD + badge

**Goal:** PR verified, merged, badge posted, result on the Arena. Satisfies "CI/CD ran".

### Tasks
- [ ] Open the submission PR via HF CLI (`--create-pr`) to the DF dataset repo.
- [ ] Confirm `verify-hf-pr.yml` fires and posts a ✅ verdict on the HF discussion.
- [ ] Maintainer fills `reproduction:` (reproduced_by/at, bench_version, `match: scoring`),
      merge the PR.
- [ ] Confirm `post-merge-badge.yml` fires and posts the badge comment.

### Verification
- [ ] CI verdict table is ✅ on the PR discussion.
- [ ] Badge comment present after merge (no duplicate sentinel).
- [ ] `random-baseline` row for ASVspoof2021_DF visible on the Arena (Per-dataset tab),
      coverage stays 1.0 (still gold).

**[MANUAL CHECKPOINT]** — verify badge + Arena row.

## Phase 8 — Docs fix

**Goal:** Docs match `validate.py` D3 behavior.

### Tasks
- [ ] `developing/new-dataset.md`: correct the D3 row + Step-3 note — D3 spot-checks the
      **first row's** sr + duration; whole-set duration is the builder's responsibility
      (recommend ensuring row 0 ≥ 1.0 s). Keep the "drop sub-second clips" tip as optional.
- [ ] `architecture/submission-lifecycle.md`: same correction in the D3 table row.
- [ ] Commit in the package repo. (No version bumps — data-only change.)

### Verification
- [ ] Docs D3 wording matches `validate.py:181-193` (first-row check).

## Risks & Mitigations

- **Long build / disk** — run Phase 2 in background; 4.7 TB free is ample; rely on asserts.
- **Webhook ordering** — push dataset → update manifest (refresh) → then the PR; if CI
  doesn't fire, force an Arena refresh.
- **CI secrets must be live** (`HF_BOT_TOKEN`, `GH_PAT`) or verify/badge silently print
  instead of post — provisioned per `project_phase8_secrets`.
- **LFS push** of ~80 shards may be slow — expected.

## Success Criteria (DoD)

1. `ASVspoof2021_DF` uploaded to HF; online validate all-green.
2. In manifest `core_set`; displayed on the Arena.
3. `random-baseline` has a reproduced, merged result on it.
4. CI/CD loop ran (verify-pr ✅ + post-merge badge); badge + result visible.
