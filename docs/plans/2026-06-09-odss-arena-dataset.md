# Arena dataset plan: ODSS

---

# Plan (reviewed at the 🚦 PLAN GATE — before any probe/build/push)

- **Dataset Name:** `ODSS`  (source casing; = `benchmarks/ODSS/`, HF repo
  `SpeechAntiSpoofingBenchmarks/ODSS`, manifest id. No lowercase slug.)
- **Raw source (read-only):** `/home/kirill/mnt/users_4tb/datasets/odss`
  - `natural/<corpus>/<speaker>/<stem>.wav`            → **bonafide** (7961)
  - `vits/<corpus>/<speaker>/<stem>.wav`               → **spoof** (11032, VITS)
  - `fastpitch-hifigan/<corpus>/<speaker>/<stem>.wav`  → **spoof** (7961, FastPitch+HiFi-GAN)
  - `transcripts.psv` (id|text), `speaker_info.csv` (speaker,dataset,gender,language), `LICENSE`
- **Date:** 2026-06-09
- **Paper:** *An Open Dataset of Synthetic Speech* (ODSS), IEEE doc 10374863
  (https://ieeexplore.ieee.org/document/10374863/). Dataset on Zenodo:
  https://zenodo.org/records/8370669 (DOI 10.5281/zenodo.8370669). No arXiv.

## What ODSS is (provenance — per your note)
Multilingual (en/de/es), multispeaker (156 voices) **synthetic-speech detection** set:
each natural utterance is paired with TTS re-synthesis of the same text. Source speech
corpora: **VCTK** (en), **Hi-Fi TTS** (en), **HUI-audio-corpus-german** (de),
**OpenSLR-ES** (es). The card will state the corpus provenance prominently, incl.
**VCTK** (used for the English VITS voices). ⚠️ Note: ODSS is *multi-corpus*, not
exclusively VCTK-based — VCTK is one of four source corpora. `vits/vctk` (3071 spoof)
has **no** natural counterpart in this release (real VCTK is distributed separately),
so VCTK appears on the spoof side only — fine for a pooled bonafide-vs-spoof EER.

## Protocol → schema mapping
- **label** ← top-level dir: `natural/` = **bonafide**, `vits/` + `fastpitch-hifigan/` = **spoof**.
- **`utterance_id`** ← the full source-relative path (generator+corpus+speaker+stem),
  e.g. `vits__vctk__p293__p293_168`. The bare stem repeats across the 3 generators
  (natural/vits/fastpitch synthesize the *same* texts) → the generator prefix makes it
  unique. `path` = source-relative path (`vits/vctk/p293/p293_168.wav`) → also unique.
  (Cf. [[reference_multisubset_uid_collision]].)
- **`notes`** extra fields: `generator` (natural | vits | fastpitch-hifigan),
  `source_corpus` (hifi-tts | hui-acg | openslr-es | vctk), `speaker`,
  `language` (en/de/es via corpus), `attack` (= generator for spoof, "bonafide" for natural).
- **Expected counts:** total **26954** / bonafide **7961** / spoof **18993**
  (VITS 11032 + FastPitch 7961). Final counts after dropping any sub-1.0 s clip — reported post-probe.

## License & redistribution
- **CC BY-SA 4.0** — found in the source `LICENSE` file (Creative Commons
  Attribution-ShareAlike 4.0). This **resolves your Zenodo "no license" doubt**:
  redistribution + derivatives are permitted under attribution + ShareAlike (it is
  **not** an ND/NC license, so the parquet is fine — cf. [[reference_license_nd_blocks_arena]]).
- HF card: `license: cc-by-sa-4.0`; ShareAlike honored (the packaged dataset is itself
  CC BY-SA 4.0). Attribution to the ODSS authors + the four source corpora (incl. VCTK).

## Manifest placement
- **Set:** Core (default). (Core re-computes coverage for every existing submission — flagged in the PR.)

## Build approach (confirm clean vs re-encode after the whole-set probe)
- **Source SR / format:** sampled clips are already **16 kHz mono PCM_16 WAV**; durations
  span ~2–6 s. → **CLEAN raw-byte embed** (pending the whole-set decode probe). Any decode
  failure → re-encode (librosa→FLAC).
- **Shard sizing:** ~2.7 GB raw WAV across 26 954 clips → **~8 shards** (~340 MB each),
  processed sorted by `utterance_id`.
- **Reference dir to fork:** `benchmarks/ASVspoof5/` (clean raw-byte embed path).
- Drop any sub-1.0 s clip; ensure row 0 ≥ 1.0 s.

## 🚦 PLAN GATE — present the above; await explicit OK. Probe/build/push nothing before this.

---

# Execution log (filled autonomously after approval)

- [x] Scaffolded at `benchmarks/ODSS/`; .gitattributes/.gitignore copied; LICENSE.txt = CC BY-SA 4.0 + provenance
- [x] **Whole-set decodability probe:** 0/26954 fails, all 16 kHz mono PCM_16, min dur 2.06 s → **CLEAN raw-byte embed**, 0 dropped
- [x] Build: sorted by uid; **8 shards**, 2.5 GB; counts asserted (**7961 / 18993**); labels.parquet emitted
- [x] `validate-dataset ./ODSS --skip-submissions` → **D1–D7 green**
- [x] Pushed `SpeechAntiSpoofingBenchmarks/ODSS` @ **`1968e6d0ef141c4572073695bdc1d17a8706177f`**; online sanity (stream + labels.parquet 26954 {7961/18993}) OK
- [x] Manifest PR (`core_set` @ `1968e6d` + CHANGELOG `dataset_added`): **discussions/17**; local clone reverted
- [x] Random-baseline seeded: submission PR **discussions/1**; **EER 49.64%**, 0 skipped; scores @ random-baseline-asas 52f656a
- [x] **Maintainer half:** merged manifest #17 + reproduce (sha matched, Δ EER 0.0) + merged submission #1 + re-ingest (173 rows, 1 ODSS, 0 warnings → cache.json to Space) + verified live (RUNNING, /healthz ok, random-baseline **gold**, rank #25/25)

## Notes / open question for the user
- You said "this is a VCTK-based dataset" — strictly, ODSS is **multi-corpus** (VCTK +
  Hi-Fi TTS + HUI-de + OpenSLR-es). I'll document VCTK provenance prominently but frame it
  as one of four source corpora. Shout if you want it framed differently.
