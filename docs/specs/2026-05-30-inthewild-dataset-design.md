# In-the-Wild — dataset addition + random-baseline end-to-end

**Date:** 2026-05-30
**Status:** approved (design)
**Author:** Claude (brainstorming session)

## Goal

Add `SpeechAntiSpoofingBenchmarks/InTheWild` to the Arena as a **Core** dataset, then
run the `random-baseline` system against it, get it merged + verified, add dynamic
badges to the `random-baseline-asas` model card, and confirm it on the live Arena.

This is two sequenced sub-projects. **B depends on A's published commit SHA**, so A
must land first.

## Labor split (decided)

- Claude **authors every file** and **runs every automated/offline command and the HF
  push** (`HF_TOKEN` is present in the env): build, `validate-dataset`, push, online
  validate, `run`, the manual submit uploads, `reproduce`.
- The **user does only the merge/UI steps**: merge the `arena-manifest` PR (M1), merge
  the dataset's `submissions/random-baseline.yaml` PR (M2), and trigger any Arena
  re-ingest / refresh. Claude surfaces these as explicit checkpoints and waits.

## Source data (confirmed by inspection)

Location: `/home/kirill/mnt/users_4tb/datasets/release_in_the_wild/`

- 31,779 `.wav` files named `0.wav … 31778.wav`.
- `meta.csv` — 31,779 rows, 3 columns: `file,speaker,label`.
- `attribution.txt` — "We thank 'VocalSynthesis' for his audio deepfakes…".
- **License: Apache-2.0** (per deepfake-total.com/in_the_wild) → redistribution permitted.
- Audio probe (row `0.wav`): **16 kHz, mono, FLOAT subtype**, 1.82 s.

Counts (1:1 wav ↔ meta.csv row):

| Field | Distribution |
|---|---|
| label | `bona-fide` **19,963** / `spoof` **11,816** (total **31,779**) |
| speaker | public figures / politicians (names already shipped in `meta.csv`) |

Paper: arXiv **2203.16263**, "Does Audio Deepfake Detection Generalize?".

## Decisions

- **Trial scope:** all **31,779** clips; no duration filtering (keep every decodable
  clip). Primary EER over the full set.
- **Set tier:** **Core** — counts toward coverage/tiers and the global ranking. Adding to
  Core shifts everyone's coverage; deliberate, like the other four Core datasets.
- **Label map:** `bona-fide → bonafide`, `spoof → spoof` (note the source hyphen).
- **Audio encoding:** source is 16 kHz mono **FLOAT** wav, so CD-ADD's bit-exact int16
  trick does not apply. Embed as **FLAC PCM_16** by reading **float** (default dtype)
  and letting `soundfile` scale to 16-bit PCM on the FLAC *write* — reading a FLOAT wav
  with `dtype="int16"` returns all zeros (libsndfile does not scale float→int on read),
  which would silently store silence. This is a float→int16 conversion, **not**
  bit-exact, so `_verify` asserts schema/row-0/decodability/**non-silence** only (no
  bit-exact guard). No resample (already 16 kHz).
- **`utterance_id`:** `ITW_<stem>` (`ITW_0 … ITW_31778`) — unique, stable join key.
- **Shards:** **8** (~3,972 rows/shard).
- **Builder location:** **inside the dataset repo** at `benchmarks/InTheWild/build_parquet.py`
  (the canonical layout `scaffold-dataset` produces). No separate `dataset-builders/` copy.
- **License id:** `apache-2.0` in the README front-matter.
- **Spec location:** this file, in the package repo `docs/specs/`.

## A. The dataset

### A1. Repo layout — scaffold then fill

New local repo `benchmarks/InTheWild/` (scaffold via
`speech-spoof-bench scaffold-dataset --name InTheWild --output-dir benchmarks/InTheWild`,
then fill; cross-check against the `benchmarks/CD-ADD` sibling):

```
InTheWild/
├── README.md            # HF card front-matter + body (ITW-specific)
├── eval.yaml            # task block; metrics: [eer_percent]
├── LICENSE.txt          # Apache-2.0 text + VocalSynthesis attribution + paper cite
├── build_parquet.py     # the builder (A2)
├── submissions/
│   ├── README.md            # submitter instructions (id + n_trials = 31779)
│   └── results_template.yaml # dataset.id + n_trials: 31779
├── data/                # generated: test-*.parquet + labels.parquet
├── .gitignore
└── .gitattributes       # *.parquet filter=lfs
```

Canonical 4-column parquet schema (validator-enforced):
`path: string`, `audio: Audio(16000)`, `label: ClassLabel(["bonafide","spoof"])`,
`notes: string`.

`notes` JSON per row: `{"utterance_id", "speaker", "label"}` (the source label string
kept for traceability; `speaker` is the public-figure name from `meta.csv`).

README front-matter must contain every D6 key
(`license, language, pretty_name, task_categories, size_categories, configs, tags, arxiv`)
and `tags` must include `arena-ready`. Values: `license: apache-2.0`, `language: [en]`,
`pretty_name: "In-the-Wild Audio Deepfake Dataset"`,
`task_categories: [audio-classification]`, `size_categories: [10K<n<100K]`,
`arxiv: ["2203.16263"]`.

### A2. `build_parquet.py` design

Mirrors `benchmarks/CD-ADD/build_parquet.py`. Constants:
`SRC_ROOT=/home/kirill/mnt/users_4tb/datasets/release_in_the_wild`,
`META_PATH=SRC_ROOT/meta.csv`, `NUM_SHARDS=8`, `EXPECTED_ROWS=31779`,
`EXPECTED_BONAFIDE=19963`, `EXPECTED_SPOOF=11816`, `TARGET_SR=16000`.
`FEATURES` = the canonical 4-column `Features`.

Steps:

1. **Parse `meta.csv`** → one record per wav: `path="<n>.wav"`, `label` mapped,
   `utterance_id=f"ITW_{stem}"`, `notes=json({utterance_id, speaker, label_str})`.
   Assert 31,779 rows + label counts in full mode.
2. **Decode filter:** full `soundfile.read` per clip; keep decodable, drop+report
   undecodable (CD-ADD pattern). Sort by numeric stem for determinism; swap a clip with
   duration ≥ 1.0 s to index 0 (D3 only checks row 0) without removing any short clip.
3. **Encode + assemble:** `flac_bytes(wav)` = `sf.read(wav)` (float) →
   `sf.write(buf, format="FLAC")` (scales to PCM_16 on write); embed `{"bytes": …, "path": …}`.
   `Dataset.from_generator(row_gen, features=FEATURES)`, then write 8 shards
   `test-000NN-of-00008.parquet` via `ds.shard(...).to_parquet(...)`.
4. **`data/labels.parquet`** auto-emitted at end (fast-path label verification);
   or backfill with `speech-spoof-bench emit-labels benchmarks/InTheWild`.
5. **`_verify`:** total rows == 31,779; bonafide == 19,963; spoof == 11,816; uid + path
   uniqueness; shard-0 columns == `{path,audio,label,notes}`; shard-0 row-0 decodes via
   `soundfile` at 16 kHz and ≥ 1.0 s. (No bit-exact guard — see encoding decision.)

Sample mode (`--limit N` / `ITW_BUILD_LIMIT`): single shard, skip count asserts — for the
fast offline `validate-dataset` pass before a full build.

### A3. Validate → publish → pin

1. Sample build → `validate-dataset ./benchmarks/InTheWild --skip-submissions` until
   D1–D7 green offline.
2. Full build (~31,779 rows, 8 shards) → local `validate-dataset --skip-submissions`.
3. Push to `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/InTheWild` with `HF_TOKEN`.
4. Online `validate-dataset SpeechAntiSpoofingBenchmarks/InTheWild --skip-submissions` → green.
5. Record the **commit SHA**.
6. Edit `arena-manifest/manifest.yaml` `core_set` to add the dataset at that SHA
   (lowercase hex 7–40 chars) + a `dataset_added` event in `arena-manifest/CHANGELOG.yaml`
   + a "re-ingest to subscribe InTheWild for webhook routing" note. **Data** change → no
   `schema_version` / `ranking_version` bump. Open the PR.
7. **🛠 M1 — user merges the manifest PR**, then triggers re-ingest if the Arena needs it.

## B. random-baseline end-to-end (after A lands)

The `random-baseline` system (slug `random-baseline`, model repo
`SpeechAntiSpoofingBenchmarks/random-baseline-asas`, meta at workspace-root
`random-baseline-meta.yaml`) is already merged on the other four Core datasets. Target:
`random-baseline` merged + verified on InTheWild, badged, visible on the Arena.

1. `speech-spoof-bench run --model-module <baseline-module> --datasets
   SpeechAntiSpoofingBenchmarks/InTheWild --output-dir ./results`
   → `results/InTheWild/{scores.txt,result.yaml}`; EER ≈ 50%, `n_skipped` ≈ 0.
   (Exact baseline module id located at implementation time.)
2. **Manual submit path** (reuses computed scores; avoids re-streaming audio — the path
   `submissions/README.md` documents):
   a. `hf upload SpeechAntiSpoofingBenchmarks/random-baseline-asas
      results/InTheWild/scores.txt
      .eval_results/SpeechAntiSpoofingBenchmarks/InTheWild/scores.txt --repo-type model`;
      read back the model commit SHA.
   b. Copy `submissions/results_template.yaml` → author `submissions/random-baseline.yaml`
      (schema v4): merge `random-baseline-meta.yaml`'s `system` block; `dataset.revision`
      = InTheWild pinned SHA; `scores` (`eer_percent`, `n_trials=31779`, `n_skipped`);
      `artifact.scores_url` = commit-pinned `…/resolve/<model-sha>/…`; `scores_sha256` =
      `sha256sum scores.txt`; `bench_version`; `reproduction: {}`.
   c. `speech-spoof-bench reproduce ./submissions/random-baseline.yaml --scoring --no-local`
      → green (recomputes EER within 1e-6, checks coverage `len+n_skipped==31779`).
   d. `hf upload SpeechAntiSpoofingBenchmarks/InTheWild random-baseline.yaml
      submissions/random-baseline.yaml --repo-type dataset --create-pr`.
3. **🛠 M2 — user merges the submission PR.** Post-merge-badge CI produces the paste
   comment.
4. **Badge:** add the **dynamic** tier + rank shields endpoints (slug `random-baseline`)
   to the `random-baseline-asas` model card and upload it with `HF_TOKEN`:
   ```markdown
   [![arena tier](https://img.shields.io/endpoint?url=https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/random-baseline/tier.json)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=random-baseline)
   [![arena rank](https://img.shields.io/endpoint?url=https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/random-baseline/rank.json)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=random-baseline)
   ```
5. **Arena:** confirm `random-baseline` shows InTheWild coverage; trigger a re-ingest if
   the row is stale.

## Risks / notes

- **Float→int16 conversion** (encoding decision) is a deliberate, documented loss of
  float precision; standard for 16-bit-PCM anti-spoofing. If lossless-float is later
  preferred, switch `flac_bytes` to embed original WAV bytes and re-pin.
- **`random-baseline` reproducibility:** scores deterministic given the dataset's
  utterance ids; reproduce must match within 1e-6 and coverage must equal 31,779.
- **Re-pinning:** if labels/shards change after publish, bump the manifest `revision` to
  the new SHA + a `dataset_repin` changelog note. Old submissions stay reproducible.
- **Arena package pin:** no package schema/logic change here, so `arena/requirements.txt`
  need not be bumped for this work.

## Out of scope

- The `--inference` verification level (re-running the checkpoint) — `NotImplementedError`.
- Any package code change (new metric/solver) — `eer_percent` is already registered.
- Bringing the other four datasets' random-baseline submissions to any new state.
