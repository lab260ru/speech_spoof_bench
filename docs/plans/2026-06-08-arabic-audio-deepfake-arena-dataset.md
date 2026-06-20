# Arena dataset plan: ArabicAudioDeepfake

---

# Plan (reviewed at the 🚦 PLAN GATE — before any probe/build/push)

- **Dataset Name:** `ArAD`  (card abbreviation, chosen at the gate; used verbatim for
  `benchmarks/ArAD/`, HF repo `SpeechAntiSpoofingBenchmarks/ArAD`, and the manifest id.
  Source HF repo is `DeepFake-Audio-Rangers/Arabic_Audio_Deepfake`. **Test split only.**)
- **Raw source (read-only):** `/home/kirill/mnt/users_4tb/datasets/arabic_df`
  - audio: `wav/test_*.wav` (clean **PCM_16** 16 kHz mono, indices 0–3912) — the build source
  - labels: `meta.csv` (`audio_path,label`); test rows = `new_audio/wav/0..3912.wav` (first 3913 rows)
- **Date:** 2026-06-08
- **Source link (no paper):** https://huggingface.co/datasets/DeepFake-Audio-Rangers/Arabic_Audio_Deepfake

## What "test subset" means here (verified)
- HF card declares splits **train=15648, test=3913**. Split is encoded in the local filenames:
  `wav/test_*.wav` (3913) vs `wav/train_*.wav` (15648); total 19561.
- `new_audio/wav/N.wav` (FLOAT re-export, labelled by `meta.csv`) is **positionally identical**
  to `wav/test_N.wav` for N=0..3912 (waveform maxabsdiff = 0; verified test_0/test_3912;
  train_0 ↔ new_audio/3913). So **test = first 3913 meta.csv rows** = `wav/test_0..3912.wav`.
- Building from the **PCM_16 `wav/test_*`** files sidesteps the FLOAT-wav int16 zero-read gotcha
  entirely (we never touch `new_audio/`). Labels are joined by index from `meta.csv`.

## Protocol → schema mapping
- id column → `utterance_id`: the test index N, id = `ArAD_test_{N:05d}` (audio file `wav/test_N.wav`)
- label column → bonafide/spoof: **POLARITY IS INVERTED.** HF `ClassLabel{'0': fake, '1': real}`.
  Arena `ClassLabel[bonafide=0, spoof=1]`. Therefore:
  - meta label `0` (fake)  → **spoof**
  - meta label `1` (real)  → **bonafide**
- extra `notes` fields to keep: `source_label` (raw 0/1), `split: "test"`, `lang: "ar"`
- **Expected counts:** total **3913** / bonafide (real) **532** / spoof (fake) **3381**

## License & redistribution
- SPDX / HF tag: **ODC-By 1.0** → `odc-by`. Redistribution permitted with attribution
  (same family as the ASVspoof5 reference dir). Ship verbatim `LICENSE.txt`.
- No paper / no DOI. D6 only requires the `arxiv` *key* present (value unvalidated) → `arxiv: []`;
  the HF source link goes in front-matter (`homepage:`) and a README "Source" section.

## Manifest placement
- **Set: Core (default).** ⚠️ Core re-computes coverage for **every existing submission** —
  every model loses one dataset of coverage until re-evaluated. If you'd rather this niche
  single-language / single-generator (RVC v2) set not gate tiers, say "Extended" at the gate.

## Build approach (confirm clean vs re-encode after the whole-set probe)
- Source SR / format: **16 kHz mono PCM_16 WAV**, ≤3 s clips (row 0 = 3.0 s). Decode-probe all
  3913 `wav/test_*` clips; drop any <1.0 s (task minimum) and re-derive counts if so.
- Path: **CLEAN raw-byte embed** (pending whole-set probe == 0 failures).
- Shard sizing: 3913 × ~96 KB ≈ **~360 MB → 1 shard** `data/test-00000-of-00001.parquet`
  (+ `data/labels.parquet`). Reference dir to fork: `benchmarks/ASVspoof5/` (clean-embed path).

## 🚦 PLAN GATE — present the above; await explicit OK. Probe/build/push nothing before this.

---

# Execution log (filled autonomously after approval)

- [x] Scaffolded at `benchmarks/ArAD/`; LICENSE/.gitattributes/.gitignore copied from ASVspoof5
- [x] **Whole-set decodability probe:** 0/3913 fail → path = **CLEAN**; 343 sub-1s dropped → 3570
- [x] Read params: 32 workers, sorted by uid; build ~27 s; 1 shard; 292 MB; counts asserted (484/3086)
- [x] `validate-dataset ./ArAD --skip-submissions` → D1–D7 all green
- [x] Pushed `SpeechAntiSpoofingBenchmarks/ArAD` @ `350184966eeb5b46ff2acdabd8f4d12e41e582da`;
      online sanity OK (stream rows 16kHz/3s; labels.parquet 3570 = 484+3086; polarity verified)
- [x] Manifest PR (`core_set` @ that sha + CHANGELOG `dataset_added`):
      https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/arena-manifest/discussions/13; clone reverted
- [x] Random-baseline seeded (via `submitting-arena-model`): submission PR
      https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ArAD/discussions/1; EER 49.84%
      (reproduce --scoring Δ 0.0; scores @ random-baseline-asas commit cf99a8f0)

## Maintainer to-dos (surface in final report)
- Review/merge the manifest PR (Core changes everyone's coverage)
- Re-ingest to subscribe the webhook
- Merge the random-baseline submission + fill its reproduction block (verify-pr routes only after re-ingest)

## Notes / guideline discrepancies
- Label polarity inversion (HF 0=fake/1=real → spoof/bonafide) is the highest-risk step; asserted
  via counts (3381 spoof / 532 bonafide) and a spot audio check.
