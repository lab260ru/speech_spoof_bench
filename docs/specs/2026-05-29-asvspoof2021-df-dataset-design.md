# ASVspoof2021_DF Dataset — Design

**Date:** 2026-05-29
**Status:** Approved (brainstorming)
**Goal:** Add `SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF` to the organization, get it
showing on the Arena, run the `random-baseline-asas` model against it, and land its result
through the CI/CD loop. Mirror the existing `ASVspoof2019_LA` dataset.

## Definition of Done

1. Dataset `SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF` is uploaded to HF and validates
   all-green (D1–D7) **online**.
2. It is in the Arena manifest `core_set` and displayed on the Arena.
3. The `random-baseline` system has a reproduced, merged result on it.
4. The CI/CD loop (webhook → verify-pr → post-merge-badge) ran, and the badge / result is
   visible.

## Decisions (locked during brainstorming)

| Decision | Choice | Why |
|----------|--------|-----|
| Dataset name | `ASVspoof2021_DF` | Source data on disk is the ASVspoof 2021 **DeepFake** track (`DF_E_*` ids, `ASVspoof2021.DF.cm.eval.trl.txt`), not LA. Name must match the data. |
| Scope | Full eval set, **611,829** utterances | Matches the pooled DF eval EER reported in the literature. |
| Sub-1.0 s clips (~6.7%) | **Keep all** | Faithful to the official protocol. D3 only spot-checks the first row, so we guarantee row 0 ≥ 1.0 s and keep everything else. |
| Manifest tier | `core_set` | Flagship ASVspoof dataset; the only existing system (`random-baseline`) gets a DF result immediately, so its coverage stays 1.0 (gold). |
| Version bumps | **None** | Adding a dataset is a *data* change (versioning.md trigger matrix). No package/schema/ranking/arena-pin bump. Manifest commit + CHANGELOG note only. |

## Source data

`/home/kirill/mnt/users_4tb/datasets/asvspoof2021_DF/`

- `ASVspoof2021_DF_eval/flac/` — 611,829 `.flac`, all 16 kHz, 34 GB.
- `ASVspoof2021_DF_eval/ASVspoof2021.DF.cm.eval.trl.txt` — eval trial list (ids only).
- `ASVspoof2021_DF_eval/LICENSE.DF.txt`, `README.DF.txt` — ODbL license + provenance.
- `trial_metadata.txt` — the keys file, 611,829 rows, 13 space-separated columns:

  | Col | Field | Example |
  |-----|-------|---------|
  | 1 | speaker_id | `LA_0023`, `TEF2` |
  | 2 | utterance_id | `DF_E_2000011` |
  | 3 | codec | `nocodec`, `low_m4a`, `high_mp3`, … (9 conditions) |
  | 4 | source | `asvspoof`, `vcc2018`, `vcc2020` |
  | 5 | attack_id | `A14`, `Task1-team20`, `-` (bonafide) |
  | 6 | label | `bonafide` / `spoof` |
  | 7 | trim | `notrim` |
  | 8 | subset | `progress` (59,325) / `eval` (533,928) / `hidden` (18,576) |
  | 9 | vocoder | `traditional_vocoder`, `neural_vocoder_*`, `-` |
  | 10–13 | task/team/gender/extra | `Task1 team20 FF E` or `- - - -` |

  Label distribution: 22,617 bonafide / 589,212 spoof.

## Target repo layout

On the 8 TB drive (`/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF/`),
pushed to `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF`. Mirrors
`ASVspoof2019_LA`:

```
ASVspoof2021_DF/
├── README.md            # HF card front-matter (all D6 keys + arena-ready) + citations + stats
├── eval.yaml            # antispoofing_eval; metrics: [eer_percent]
├── LICENSE.txt          # ODbL (from source LICENSE.DF.txt)
├── build_parquet.py     # raw flac + trial_metadata.txt → canonical parquet
├── protocols/
│   └── ASVspoof2021.DF.cm.eval.trl.txt   # copied for provenance
├── data/
│   └── test-*.parquet   # ~80 shards (~450 MB each)
├── submissions/
│   ├── README.md            # submitter instructions (DF paths)
│   ├── results_template.yaml # n_trials: 611829, dataset id updated
│   └── random-baseline.yaml  # filled after the run reproduces
├── tests/
│   └── test_schema.py
├── .gitattributes       # *.parquet + *.tar.gz via LFS
└── .gitignore
```

Canonical parquet schema (D1): exactly `{path, audio, label, notes}`.

## `build_parquet.py`

A **fresh raw build** (the 2019 script migrates from existing parquet; ours reads raw
flac + metadata directly).

```python
REPO_ROOT   = Path(__file__).resolve().parent
SRC_ROOT    = Path("/home/kirill/mnt/users_4tb/datasets/asvspoof2021_DF")
FLAC_DIR    = SRC_ROOT / "ASVspoof2021_DF_eval" / "flac"
META_PATH   = SRC_ROOT / "trial_metadata.txt"
PARQUET_DIR = REPO_ROOT / "data"
NUM_SHARDS  = 80
EXPECTED_ROWS = 611829

FEATURES = Features({
    "path":  Value("string"),
    "audio": Audio(sampling_rate=16000),
    "label": ClassLabel(names=["bonafide", "spoof"]),
    "notes": Value("string"),
})
```

Steps:

1. Parse `trial_metadata.txt` → list of dicts `{uid, speaker, codec, source, attack, label, subset, vocoder}`.
2. Assert every `uid` has `FLAC_DIR/<uid>.flac` and the count == `EXPECTED_ROWS`.
3. Sort by `uid` for determinism.
4. **Guarantee row 0 ≥ 1.0 s** (D3): probe with `soundfile.info` from the front, move the
   first clip whose duration ≥ 1.0 s to index 0. (Header-only read, cheap.)
5. `notes` per row:
   ```json
   {"utterance_id": "...", "speaker_id": "...", "subset": "...",
    "codec": "...", "source": "...", "attack_id": "...", "vocoder": "..."}
   ```
6. `path` = `"<uid>.flac"`, `audio` = the flac path (datasets reads + encodes), `label` =
   the bonafide/spoof string.
7. Build with `Dataset.from_generator(rows, features=FEATURES)`; shard into `NUM_SHARDS`
   `test-NNNNN-of-00080.parquet` (write to tmp, then copy into `data/`).
8. Post-write asserts: total rows == 611,829; unique uids/paths; shard-0 columns ==
   `{path, audio, label, notes}`; shard-0 row-0 sr==16000 and dur ≥ 1.0 s.

A copy lives in `dataset-builders/ASVspoof2021_DF/` per that repo's convention.

## README front-matter & eval.yaml

Front-matter (D6 keys + `arena-ready`):

```yaml
license: other            # ODbL — described in body, matches 2019's `other` convention
language: [en]
pretty_name: ASVspoof 2021 DF
task_categories: [audio-classification]
size_categories: [100K<n<1M]
configs:
  - {config_name: default, data_files: [{split: test, path: "data/test-*.parquet"}]}
tags: [anti-spoofing, audio-deepfake-detection, speech, benchmark, arena-ready]
arxiv: ["2109.00537"]
```

Body: overview, ODbL license note, schema table, stats (611,829 / 22,617 / 589,212),
source provenance, evaluation pointer, **both** citations:

- arXiv `@misc{yamagishi2021asvspoof2021acceleratingprogress, … eprint=2109.00537}`
- Interspeech `@inproceedings{yamagishi21_asvspoof, … doi=10.21437/ASVSPOOF.2021-8}`

`eval.yaml` — identical task block to 2019:

```yaml
name: ASVspoof 2021 DF
description: >
  DeepFake (DF) evaluation partition of ASVspoof 2021. Binary classification:
  bonafide vs. spoof. EER on the official DF eval protocol.
evaluation_framework: inspect-ai
tasks:
  - id: antispoofing_eval
    config: default
    split: test
    field_spec: {input: audio, target: label}
    solvers: [{name: speech_spoof_bench_solver}]
    scorers: [{name: speech_spoof_scorer}]
    metrics: [eer_percent]
```

## Validate → publish → manifest

1. `speech-spoof-bench validate-dataset <local-dir> --skip-submissions` → all-green D1–D7.
2. Register locally (`local set`) and smoke-test the random baseline offline.
3. Push the repo to HF; note the commit SHA.
4. `speech-spoof-bench validate-dataset SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF`
   (online) → confirm green against what HF serves.
5. **arena-manifest** PR/commit: add to `core_set` with the pinned SHA, add a
   `CHANGELOG.yaml` `dataset_added` entry. Pushing the manifest (an org dataset repo)
   fires the webhook → Arena re-ingests → the new repo becomes "subscribed" for routing.

```yaml
core_set:
  - {id: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA, revision: 9b2040e8c57749dcd9a4f16ad61b4f47626b89ec}
  - {id: SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF, revision: <new-sha>}
```

## Run model + submission (rest of DoD)

1. Run the baseline behind `random-baseline-asas`:
   ```bash
   speech-spoof-bench run \
     --model-module speech_spoof_bench.examples.random_baseline:RandomBaseline \
     --datasets SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF --output-dir ./results
   ```
   Expect `eer_percent ≈ 50`, `n_skipped` ≈ 0, `n_trials = 611829`.
2. Upload `scores.txt` to the model repo under
   `.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF/scores.txt`; note the commit SHA.
3. Author `submissions/random-baseline.yaml` (slug `random-baseline`, pinned `scores_url`,
   `scores_sha256`, `n_trials: 611829`, dataset `revision` = the new SHA), open the PR via
   HF CLI.
4. Mirror CI locally: `speech-spoof-bench reproduce <submission>.yaml --scoring --no-local`.
5. The PR triggers `verify-hf-pr.yml`; maintainer fills `reproduction:` and merges;
   `post-merge-badge.yml` posts the badge. Result appears on the Arena.

## Docs / package fixes

- **Fix docs:** `developing/new-dataset.md` and `architecture/submission-lifecycle.md`
  describe D3 as a per-clip "drop sub-second clips" / "duration ≥ 1.0 s" guarantee, but
  `validate.py` only checks **the first row**. Correct the wording to say D3 spot-checks
  the first row's sample rate and duration (so datasets must ensure row 0 ≥ 1.0 s), and
  note that whole-set duration is the builder's responsibility.
- **No version bumps** anywhere (data-only change; package logic untouched).

## Risks / watch-items

- **Build cost:** 611k file reads + re-encode into ~36 GB parquet. Long build; 4.7 TB free
  on the target drive is ample. Run in background, verify row counts.
- **Webhook subscription ordering:** the new dataset only routes after the Arena ingests
  the updated manifest. Push dataset → update manifest (triggers refresh) → then the
  submission PR. If CI doesn't fire, force an Arena refresh.
- **CI secrets** (`HF_BOT_TOKEN`, `GH_PAT`, …) must be live for verify/badge to post
  rather than print. These are provisioned (project memory `project_phase8_secrets`).
- **LFS:** ~80 large parquet shards via `.gitattributes` LFS — push may be slow.
