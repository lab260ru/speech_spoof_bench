# Arena dataset plan: LibriSeVoc

# Plan (reviewed at the 🚦 PLAN GATE — before any probe/build/push)

- **Dataset Name:** `LibriSeVoc`  (source casing; = `benchmarks/LibriSeVoc/`, HF repo `SpeechAntiSpoofingBenchmarks/LibriSeVoc`, manifest id; no lowercase slug)
- **Raw source (read-only):** audio dir `/home/kirill/mnt/users_4tb/datasets/librivoc/LibriSeVoc/` (subdirs `gt/` + 6 vocoders); protocol/split file `/home/kirill/mnt/users_4tb/datasets/librivoc/test.txt`
- **Date:** 2026-06-07
- **Subset:** **test split only** (the 2641 base utterances listed in `test.txt`; dev/train ignored)

## Source layout (verified)
`LibriSeVoc/` holds 7 sibling dirs, each with 13201 clips (all `.wav`, 24 kHz mono):
- `gt/` — genuine LibriTTS audio → **bonafide**  (`<base>.wav`)
- `diffwave/ melgan/ parallel_wave_gan/ wavegrad/ wavenet/ wavernn/` — re-synthesised by 6 vocoders → **spoof** (`<base>_gen.wav`)

`test.txt` lists 2641 unique base filenames (e.g. `6147_34605_000006_000009.wav`); train/dev/test partition the 13201 utterances with no overlap. Every test base exists as 1 bonafide + 6 vocoder clips.

## Protocol → schema mapping
- id → `utterance_id`: synthesised, unique & stable — `gt_<base>` for bonafide, `<vocoder>_<base>` for spoof (the raw base repeats across all 7 dirs, so it can't be the id alone)
- `path` (unique): the source-relative path — `gt/<base>.wav` or `<vocoder>/<base>_gen.wav`
- label: directory-as-label — `gt` ⇒ bonafide, the 6 vocoder dirs ⇒ spoof
- extra `notes` fields: `speaker_id` (= base before first `_`), `vocoder` (`gt`/`diffwave`/…), `base_id`, `subset` (`test`)
- **Expected counts:** total **18487** / bonafide **2641** (gt) / spoof **15846** (2641 × 6 vocoders)

## License & redistribution
- SPDX / HF tag: **`cc-by-sa-4.0`** (user-confirmed); redistribution permitted: **yes** — CC BY-SA 4.0 allows sharing/adaptation with attribution + share-alike. Ship verbatim `LICENSE.txt`. Paper: arXiv **2304.13085** ("AI-Synthesized Voice Detection Using Neural Vocoder Artifacts", Sun et al.).

## Manifest placement
- **Set:** Core (default). Adding to Core re-computes coverage for every existing submission — flagged as a maintainer to-do.

## Build approach (re-encode — driven by sample-rate, confirmed-decodable)
- Source SR / format: **24 kHz mono WAV**, mixed `PCM_16` / `FLOAT` subtypes (e.g. `diffwave` is FLOAT); clip durations multi-second (sample row ≈ 9.8 s).
- Path: **RE-ENCODE** (not raw-embed). Canonical store is 16 kHz; raw 24 kHz bytes would (a) make `_verify`'s row0 `sr==16000` check fail and (b) store a non-canonical SR. Stage-1 `librosa.load(sr=16000, mono=True)` → clean 16 kHz mono FLAC handles the 24→16 resample **and** the FLOAT-subtype files correctly (float read, integer FLAC write). Add a **non-silence guard** in the re-encode worker (per the FLOAT-wav-zeros gotcha) so any all-zero decode aborts the build.
- Whole-set decode probe still runs (skill rule) to confirm 0 failures before committing.
- Read order: stage-1 list sorted by source path so each dir is read near-sequentially (HDD).
- Shard sizing: ~2641×7 clips of 16 kHz FLAC ≈ ~2 GB → **8 shards** (~250–300 MB each); fork `benchmarks/ASVspoof2021_LA/build_parquet.py` (the re-encode reference).

## 🚦 PLAN GATE — present the above; await explicit OK. Probe/build/push nothing before this.

---

# Execution log (filled autonomously after approval)

- [x] Scaffolded at `benchmarks/LibriSeVoc/`; LICENSE (CC BY-SA 4.0 full text)/.gitattributes/.gitignore copied
- [x] **Whole-set decode gate folded into Stage 1 re-encode:** 0/18487 failures (the standalone sf.read probe was dropped — Stage 1 decodes every clip and aborts on any failure + non-silence guard; saves re-reading 51 GB off the source HDD twice)
- [x] Read params: 64 workers, sorted by uid; Stage 1 re-encode ~2m09s + Stage 2 8 shards ~3s; 8 shards / 3.0 GB; counts asserted (18487 / 2641 / 15846)
- [x] `validate-dataset ./LibriSeVoc --skip-submissions` → D1–D7 green
- [x] Pushed `SpeechAntiSpoofingBenchmarks/LibriSeVoc` @ `13d69268a57afc7ebba2421d2a31730a390f1ff2`; online sanity (stream rows @16 kHz + labels.parquet 18487/2641/15846) OK
- [x] Manifest PR (`core_set` @ `13d69268…` + CHANGELOG `dataset_added`): https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/arena-manifest/discussions/5 ; local clone reverted
- [x] Random-baseline seeded: scores @ random-baseline-asas@e41c2663, EER **49.74%** (0 skipped, reproduce-consistent); submission PR https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/LibriSeVoc/discussions/1 (verify-pr deferred until maintainer ingest)

## Maintainer to-dos — DONE (acting as korallll, 2026-06-07)
- [x] Merged manifest PR discussions/5 → LibriSeVoc live in Core (7 datasets)
- [x] Re-ingest fired via webhook; LibriSeVoc subscribed; Space cache.json refreshed
- [x] `ci verify-pr` ✅ all checks passed (sha256 matched; EER recomputed Δ 0.0); verdict posted to PR
- [x] Filled reproduction block (match: scoring) + merged submission PR discussions/1
- [x] Confirmed LIVE: Space cache.json carries the LibriSeVoc random-baseline row (EER 49.7413); live HTML renders LibriSeVoc

## Notes / guideline discrepancies
- Skill's clean-vs-re-encode rule is decodability-only; here re-encode is chosen for the 24→16 kHz resample even though the source decodes cleanly. Noting in case the skill should mention SR-mismatch as a second re-encode trigger.
