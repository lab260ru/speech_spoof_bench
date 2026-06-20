# Arena dataset plan: J-SPAW_LA

---

# Plan (reviewed at the 🚦 PLAN GATE — before any probe/build/push)

- **Dataset Name:** `J-SPAW_LA`  (source casing `J-SPAW` + `_LA` track suffix, mirroring
  `ASVspoof2021_LA`. Used verbatim for `benchmarks/J-SPAW_LA/`, HF repo
  `SpeechAntiSpoofingBenchmarks/J-SPAW_LA`, and the manifest id. The `_LA` suffix is honest:
  this release physically contains **only** the LA track; the PA replay audio — J-SPAW's main
  contribution, 96 000 clips — is **absent** from `j-spaw_ver1`, so PA cannot be packaged now.)
- **Raw source (read-only):** `/home/kirill/mnt/users_4tb/datasets/j-spaw_ver1/wav`
  - spoof audio: `wav/LA/*.wav` (1600 clips)
  - bonafide audio: `wav/ASV/*_M2_*{BT,BU,BV,BW,BX}.wav` (800 clips — the LA bonafide eval set)
  - protocol: `wav/metadata_LA.txt` (2400 rows; ASVspoof-2021-style columns per `wav/config.py`
    `ConfigLA`: `spk trial codec trans attack label trim subset`)
- **Date:** 2026-06-09
- **Paper:** Shiota et al., "J-SPAW: Japanese speaker verification and spoofing attacks recorded
  in-the-wild dataset", Interspeech 2025. DOI `10.21437/Interspeech.2025-352`. No arXiv.
  Japanese-language, in-the-wild PA-focused corpus (this packaging = the auxiliary "(LA)" track).

## Protocol → schema mapping
- **id column → `utterance_id`** (= audio filename stem): protocol col 2 `trial`
  (e.g. `F001_R1_E2_L1_BT` spoof, `F001_R1_E2_M2_BT` bonafide). Verified unique across all 2400
  (spoof carry `L1/L2`, bonafide carry `M2` → no collision). `path` = original relative path
  (`LA/<stem>.wav` vs `ASV/<stem>.wav`) → also unique.
- **label column → bonafide/spoof:** protocol col 6 `label` (`bonafide` | `spoof`).
- **extra `notes` fields to keep:** `speaker` (col1), `track` = `"LA"`, `attack` (col5:
  `L1`/`L2` for spoof, `bonafide` for genuine), `record_env` (col4 `trans`: `E1`–`E4` =
  quiet / air-con / music / outside), `mic` (`M2` for bonafide; loudspeaker for spoof),
  `subset` (col8, all `eval`). Enables the per-condition (attack × env) analysis the paper emphasises.
- **Expected counts:** total **2400** / bonafide **800** / spoof **1600**
  (spoof split L1=800, L2=800; each label evenly across E1–E4).

## License & redistribution
- **⚠️ NON-COMMERCIAL USE ONLY.** Source license (github.com/takamichi-lab/j-spaw) is a bare
  *"For non-commercial use only"* — no explicit redistribution or derivatives clause.
- HF card: `license: other`, `license_name: "non-commercial-research-only"`,
  `license_link:` the J-SPAW GitHub README. The card README will carry a **prominent ⚠️ banner**
  highlighting the non-commercial restriction (per your request to highlight it).
- Redistribution basis: packaging into a non-commercial research benchmark is consistent with the
  stated NC restriction; you are directing this with awareness of the sparse terms. Clean raw-byte
  embed (no re-encode) keeps it a verbatim copy, not a derivative — the most conservative form.

## Manifest placement
- **Set:** Core (default). (Core re-computes coverage for every existing submission — expected;
  will be called out in the manifest PR.)

## Build approach (confirm clean vs re-encode after the whole-set probe)
- **Source SR / format:** already **16 kHz mono PCM_16 WAV** (this `ver1` release is pre-resampled
  from the paper's 48 kHz capture). Sampled durations ≥ ~1.17 s (well over the 1.0 s floor).
- **Path:** **CLEAN raw-byte embed** (source is already the canonical 16 kHz mono) — pending the
  whole-set soundfile decode probe. Any decode failure → fall back to re-encode (librosa→FLAC).
- **Shard sizing:** ~170 MB total (112 MB spoof + 58 MB bonafide) → **single shard**
  `data/test-00000-of-00001.parquet` + `data/labels.parquet`.
- **Reference dir to fork:** `benchmarks/ASVspoof5/` (clean raw-byte embed path).
- Drop any sub-1.0 s clip if the whole-set probe surfaces one; ensure row 0 ≥ 1.0 s.

## 🚦 PLAN GATE — present the above; await explicit OK. Probe/build/push nothing before this.

---

# Execution log (filled autonomously after approval)

- [x] Scaffolded at `benchmarks/J-SPAW_LA/`; .gitattributes/.gitignore copied; LICENSE.txt = NC terms
- [x] **Whole-set decodability probe:** 0/2400 fails, all 16 kHz mono PCM_16 → path = **CLEAN raw-byte embed**
- [x] 3 sub-1.0s clips dropped (all spoof/L1) → **2397 / 800 bonafide / 1597 spoof** asserted; 1 shard, 163 MB; + labels.parquet
- [x] `validate-dataset ./J-SPAW_LA --skip-submissions` → **D1–D7 all green**
- [x] Pushed `SpeechAntiSpoofingBenchmarks/J-SPAW_LA` @ **`b900644bf4af592f8cbb2508a61d05e9849ca6dd`**; online sanity (stream decode + labels.parquet 2397 {800/1597}) OK
- [x] Manifest PR (`core_set` @ `b900644` + CHANGELOG `dataset_added`): **discussions/16**; local clone reverted clean
- [x] Random-baseline seeded (via `submitting-arena-model`): submission PR **discussions/1**; EER **50.5%** (recompute Δ0.015), n_skipped 0; scores pinned to random-baseline-asas @ 9b0e0c7

## Maintainer steps — DONE (user said "make it by yourself")
- [x] Merged manifest PR **#16** → J-SPAW_LA in `core_set` on main
- [x] `reproduce --scoring` gate: sha256 matched, Δ EER 0.0 (50.5 vs 50.5), 2397 trials / 0 skipped
- [x] Filled `reproduction: {match: scoring}` on PR #1 branch → merged submission PR **#1**
- [x] Re-ingest: `load_state(force_refresh=True)` → 164 rows, **1 J-SPAW_LA row, 0 warnings** → committed `cache.json` to the Space
- [x] Verified live: Space RUNNING, `/healthz` ok, `random-baseline` tier=**gold** (coverage 1.0 ⇒ J-SPAW_LA counted), rank #21/21

## Notes / guideline discrepancies
- PA track absent from `ver1` (96 000 replay clips + their 800-clip bonafide eval set not shipped);
  only the LA track is packageable. Named `J-SPAW_LA` to leave room for a future `J-SPAW_PA`.
- Source `metadata_PA.txt` (96 800 rows) and `metadata_LA.txt` bonafide rows reference files that,
  for PA, are not present — confirmed LA's 2400 rows all resolve to on-disk audio.
