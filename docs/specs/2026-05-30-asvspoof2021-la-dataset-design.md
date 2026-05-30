# ASVspoof2021_LA — dataset addition + random-baseline end-to-end

**Date:** 2026-05-30
**Status:** approved (design)
**Author:** Claude (brainstorming session)

## Goal

Add `SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA` to the Arena as a **Core** dataset,
then run the `random-baseline` system against it (and bring the existing DF / 2019_LA
random-baseline submissions to *merged + verified*), add badges, and confirm everything
on the live Arena. While executing, audit `docs/developing/new-dataset.md` for
correctness and record every stale/incorrect instruction for a follow-up doc fix.

This is two sequenced sub-projects. **B depends on A's published commit SHA**, so A
must land first.

## Labor split (decided)

- Claude **authors every file** (dataset repo contents, build script, manifest/changelog
  edits, submission YAMLs) and produces a copy-paste **command runbook**.
- The **user runs every command that touches data or Hugging Face** (sample build, full
  build, validate-dataset, push, run, submit, reproduce, badge). Claude runs nothing that
  touches the data or HF; the user pastes back outputs (commit SHAs, EER, PR URLs) and
  Claude folds them into the authored files.

## Source data (confirmed by inspection)

Location: `/home/kirill/mnt/users_4tb/datasets/ASVspoof2021_LA_eval/`

- `flac/` — 181,566 `.flac` files, already 16 kHz / 16-bit.
- `trial_metadata.txt` — 181,566 rows, **8 space-separated columns**:
  `speaker_id  utterance_id  codec  transmission  attack_id  label  trim  phase`
- `ASVspoof2021.LA.cm.eval.trl.txt` — 181,566 ids (the eval protocol list).
- `LICENSE.txt` — **ODC-By** (Open Data Commons Attribution) → redistribution permitted.
- `README.LA.txt` — confirms 16 kHz, ODC-By.

Counts (1:1 flac ↔ metadata ↔ trl, all keyed):

| Field | Distribution |
|---|---|
| label | bonafide **18,452** / spoof **163,114** (total **181,566**) |
| codec | alaw, ulaw, gsm, pstn, g722, opus, none (7) |
| attack_id | A07–A19 + `bonafide` |
| phase | eval 148,176 / progress 16,464 / hidden 16,926 |
| transmission | ita_tx, sin_tx, loc_tx, mad_tx, `-` |
| speaker | 67 unique |

Note vs DF: LA metadata has **no `vocoder`/`source` column** (DF had 9+). The LA codec
axis (telephony transmission codecs) is the headline difference from ASVspoof2019_LA.

## Decisions

- **Trial scope:** include **all 181,566** keyed trials; store `phase` in `notes` so
  anyone can sub-select. Matches how `ASVspoof2021_DF` ships (full eval set) and the
  pooled-EER number reported in the literature. Primary EER is computed over the full set.
- **Set tier:** **Core** (counts toward coverage/tiers and the global ranking, like
  2019_LA and 2021_DF). Adding to Core shifts everyone's coverage — a deliberate change.
- **Re-encode:** **probe-then-conditional** — sample-test source FLAC with `soundfile`;
  re-encode (bit-exact, DF-style) only if any fail to decode, otherwise embed raw bytes.
- **Build parallelism:** the **whole build is multiprocess** (probe + assembly across a
  process pool), not single-process.
- **Shards:** **24** (~7,565 rows/shard, matching DF's ~7,648 rows/shard density).
- **License id:** `odc-by` in the README front-matter.
- **Spec location:** this file, committed to the package repo.

## A. The dataset

### A2. Repo layout — clone the `ASVspoof2021_DF` sibling and adapt

New local repo `benchmarks/ASVspoof2021_LA/`, mirroring `benchmarks/ASVspoof2021_DF/`:

```
ASVspoof2021_LA/
├── README.md            # HF card front-matter + body (LA-specific)
├── eval.yaml            # identical task block to DF; metrics: [eer_percent]
├── LICENSE.txt          # ODC-By text copied from the source dir
├── build_parquet.py     # adapted (A3)
├── protocols/
│   └── ASVspoof2021.LA.cm.eval.trl.txt
├── submissions/
│   ├── README.md            # DF copy, id + n_trials swapped
│   └── results_template.yaml # DF copy, dataset.id + n_trials: 181566 swapped
├── tests/test_schema.py # mirrored
├── data/                # generated: test-*.parquet + labels.parquet (gitignored until built)
├── .gitignore           # __pycache__, .pytest_cache, *.pyc, _clean_flac/
└── .gitattributes       # *.parquet filter=lfs; *.tar.gz filter=lfs
```

Canonical 4-column parquet schema (validator-enforced):
`path: string`, `audio: Audio(16000)`, `label: ClassLabel(["bonafide","spoof"])`,
`notes: string`.

`notes` JSON per row:
`{"utterance_id", "speaker_id", "codec", "transmission", "attack_id", "trim", "phase"}`.

README front-matter must contain every D6 key
(`license, language, pretty_name, task_categories, size_categories, configs, tags, arxiv`)
and `tags` must include `arena-ready`. Values: `license: odc-by`,
`pretty_name: "ASVspoof 2021 LA"`, `size_categories: [100K<n<1M]`, `arxiv: ["2109.00537"]`.

### A3. `build_parquet.py` design (multiprocess, probe-then-conditional, resumable)

Constants: `SRC_ROOT=/home/kirill/mnt/users_4tb/datasets/ASVspoof2021_LA_eval`,
`FLAC_DIR=SRC_ROOT/flac`, `META_PATH=SRC_ROOT/trial_metadata.txt`, `NUM_SHARDS=24`,
`EXPECTED_ROWS=181566`, `EXPECTED_BONAFIDE=18452`, `EXPECTED_SPOOF=163114`, `TARGET_SR=16000`.
`FEATURES` = the canonical 4-column `Features`.

Steps:

1. **Parse metadata** (8-col parser → list of dicts). Assert 181,566 rows in full mode.
   Verify flac presence. Sort by `utterance_id` for determinism. Swap a clip with
   duration ≥ 1.0 s (header-only probe) to index 0 (D3 only checks row 0).
2. **Probe stage (parallel):** `soundfile.read` a random sample (~3,000 uids) across a
   `ProcessPoolExecutor`. `dirty = any failures`.
3. **Assembly (parallel — one worker per shard):** partition rows into 24 contiguous
   shards. A process pool runs one worker per shard; each worker builds its shard via
   `Dataset.from_generator(rows_for_shard, features=FEATURES).to_parquet(
   test-000NN-of-00024.parquet)` so the per-shard schema matches the validator exactly.
   - **clean path:** embed raw source FLAC bytes directly (already 16 kHz mono 16-bit).
   - **dirty path:** `librosa.load(sr=16000, mono=True)` → `soundfile.write(FLAC)` to a
     clean buffer, embed those bytes (PCM preserved bit-exactly).
   - Write each shard to a temp name then `os.replace` → **resumable** (a worker skips a
     shard whose final file already exists and is non-empty).
4. **`data/labels.parquet`** via `speech_spoof_bench.labels.emit_labels(REPO_ROOT)`.
5. **`_verify`:** total rows == 181,566; bonafide == 18,452; spoof == 163,114; uid + path
   uniqueness; shard-0 columns == `{path,audio,label,notes}`; shard-0 row-0 decodes with
   `soundfile` at 16 kHz and ≥ 1.0 s.

Sample mode (`--limit N` or env): single shard, skip count asserts — used for the fast
offline `validate-dataset` pass before a full build.

The dirty-path re-encode uses a `_clean_flac/` staging dir (gitignored) only if the probe
trips; the clean path needs no staging.

### A4. Validate → publish → pin

Commands authored for the user to run (offline first, then online):

1. Sample build → `validate-dataset ./benchmarks/ASVspoof2021_LA --skip-submissions`
   until D1–D7 green offline.
2. Full build → local `validate-dataset` (no skip needed once submissions exist; use
   `--skip-submissions` until the random-baseline PR is merged).
3. Push to `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA`.
4. Online `validate-dataset SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA` → green.
5. User reports the **commit SHA**.
6. Claude edits `arena-manifest/manifest.yaml` `core_set` to add the dataset at that SHA
   (lowercase hex 7–40 chars), plus a `dataset_added` note in `arena-manifest/CHANGELOG.yaml`.
   This is a **data** change → no `schema_version` / `ranking_version` bump.
7. User pushes the manifest; the webhook refreshes the Arena cache. Re-ingest if needed
   (a manifest `note` event can document the re-ingest, mirroring the DF rollout).

## B. random-baseline end-to-end (runbook → own plan after A)

The `random-baseline` system (slug `random-baseline`, model repo
`SpeechAntiSpoofingBenchmarks/random-baseline-asas`) already has **opened** submissions on
DF and 2019_LA with empty `reproduction: {}`. Target end state: `random-baseline` is
**merged + verified** on all three Core datasets, badged, and visible on the Arena.

For the new **2021_LA** (and to finish DF + 2019_LA):

1. `speech-spoof-bench run --model-module <baseline-module> --datasets
   SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA --output-dir ./results`
   → `results/ASVspoof2021_LA/{scores.txt,result.yaml}`; EER ≈ 50%, `n_skipped` ≈ 0.
   (Exact baseline module id located at implementation time.)
2. `speech-spoof-bench submit …` (with a `meta.yaml`) → uploads commit-pinned `scores.txt`
   to `random-baseline-asas/.eval_results/…/ASVspoof2021_LA/scores.txt` and opens the
   `submissions/random-baseline.yaml` PR (schema v4) on the dataset repo.
3. Maintainer (user): `speech-spoof-bench reproduce ./submission.yaml --scoring --no-local`
   → fills the `reproduction:` block → merges.
4. **Badge:** the static `result.yaml` projection (schema v1) + the **dynamic** tier/rank
   shields endpoint markdown, pasted into the `random-baseline-asas` model card. Prefer the
   dynamic badges so they stay honest as the board grows.
5. **Arena:** the manifest/dataset update fires the webhook → cache refresh. Confirm
   `random-baseline` appears with 2021_LA coverage; trigger a re-ingest if the row is stale.
6. **Doc audit:** record every place `docs/developing/new-dataset.md` is wrong/stale
   (e.g. command names, paths, behavior that drifted) and fix in a follow-up package-doc
   commit.

## Risks / notes

- **LA may share DF's un-decodable-FLAC issue** (same 2021 release). The probe stage
  detects it; the dirty path handles it bit-exactly. No manual decision needed.
- **`random-baseline` reproducibility:** scores are deterministic given the dataset's
  utterance ids; reproduce recomputes EER and must match the claimed value within 1e-6.
  Coverage requires `len(scores) + n_skipped == n_trials` (181,566).
- **Re-pinning:** if labels/shards change after publish, bump the manifest `revision` to
  the new SHA + a `dataset_repin` changelog note. Old submissions stay reproducible
  against the old revision.
- **Disk:** ~24 shards; staging dir only materialized on the dirty path. Build is
  resumable per shard, so a killed run continues cheaply.

## Out of scope

- The `--inference` verification level (re-running the checkpoint) — `NotImplementedError`
  today.
- Any package code change (new metric/solver) — `eer_percent` is already registered.
- Rewriting `new-dataset.md` beyond fixing concrete inaccuracies found during execution.
