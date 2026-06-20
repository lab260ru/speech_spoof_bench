# Arena dataset plan: DFADD · PyAra · TIMIT-TTS · XMAD (batch)

Four datasets requested in one batch. **Three are buildable; one (TIMIT-TTS) is blocked.**
Each is independent — own HF repo, own manifest entry, own random-baseline seed. All **Core**.

---

# Plan (reviewed at the 🚦 PLAN GATE — before any probe/build/push)

## 1) DFADD  ✅ buildable

- **Name:** `DFADD` (paper casing; user wrote "DFAAD" — the arXiv 2409.08731 dataset is **DFADD**,
  "The Diffusion and Flow-Matching Based Audio Deepfake Dataset"). `benchmarks/DFADD/`,
  HF `SpeechAntiSpoofingBenchmarks/DFADD`, manifest id same.
- **Raw source (read-only):** `/home/kirill/mnt/users_4tb/datasets/dfadd` — **eval (`test/`) split only**.
- **Protocol → schema (directory-based):**
  - `DATASET_VCTK_BONAFIDE/test/*.wav` → **bonafide** (755, VCTK)
  - `DATASET_GradTTS/test`, `DATASET_MatchaTTS/test`, `DATASET_NaturalSpeech2/test`,
    `DATASET_PflowTTS/test`, `DATASET_StyleTTS2/test` → **spoof** (600 each = **3000**)
  - `utterance_id` = `<generator>__<stem>` (e.g. `gradtts__p227_001_GradTTS`, `bonafide__p227_001`)
    — stems are shared across generators (same VCTK texts) so the generator prefix makes them unique.
  - `notes`: `generator` (vctk/gradtts/matchatts/naturalspeech2/pflowtts/styletts2), `speaker`, `attack`.
  - **Expected counts: total 3755 / bonafide 755 / spoof 3000.**
- **License:** MIT (per your note; LICENSE.txt = MIT + VCTK attribution). Redistribution: yes.
- **arxiv:** `2409.08731`.
- **Build:** 16 kHz mono, FLAC (PCM_24) + WAV (PCM_16/32), all already 16 kHz → **CLEAN raw-byte embed**
  (pending whole-set probe). ~0.3 GB → **~2 shards**. Fork `benchmarks/ASVspoof5/`.

## 2) PyAra  ✅ buildable (⚠ no paper)

- **Name:** `PyAra`. `benchmarks/PyAra/`, HF `SpeechAntiSpoofingBenchmarks/PyAra`.
- **Raw source:** `/home/kirill/mnt/users_4tb/datasets/final_dataset` (Russian deepfake set).
  - `final_dataset.tsv` cols: `path, sentence, age, gender, fake, algorithm, length`.
  - `Real/*.wav` (`fake=0`) → **bonafide** (73583); `Fake/*.wav` (`fake=1`) → **spoof** (128195).
- **Protocol → schema:** label ← `fake` col. `utterance_id` = path stem (`real_12713`, `alg_1_0` — already
  unique). `notes`: `algorithm` (alg_1..alg_5 for spoof), `gender`, `age`, `length`. `path` = tsv path.
  - **Expected counts: total 201778 / bonafide 73583 / spoof 128195.** All clips ≥3.0 s (no sub-1 s).
- **License:** CC BY-NC-SA 4.0 (NC+SA, **not** ND → parquet OK). LICENSE.txt = CC BY-NC-SA 4.0.
- **⚠ arxiv:** no paper, no DOI. D6 *requires* the `arxiv` key be present (value unchecked). Plan: set
  `arxiv:` to empty/placeholder so D6 passes; the Arena row shows with **no paper link**. If you have a
  source URL (HF/GitHub) I'll put it in the README citation. (Cf. [[reference_paper_no_arxiv]].)
- **Build:** 16 kHz mono WAV PCM_16 → **CLEAN raw-byte embed**. ~35 GB → **~100 shards** (~350 MB).
  **Proxy OFF for push** ([[reference_proxy_large_transfers]]).

## 3) TIMIT-TTS  ❌ BLOCKED — spoof-only

- Source `/home/kirill/mnt/users_4tb/datasets/timit_tts` is **entirely synthetic** (CLEAN/AUG/DTW/DTW_AUG ×
  single/multi-speaker × ~10–14 TTS engines). There is **no real/bonafide audio anywhere** in the release
  (the original VidTIMIT/TIMIT speech is not redistributed here).
- **EER needs both classes** — a spoof-only dataset produces no meaningful EER and cannot be ranked/seeded.
  → **Cannot add as-is.** Options for you (see question).

## 4) XMAD  ✅ buildable (large)

- **Name:** `XMAD` (XMAD-Bench, arXiv 2506.00462, Cross-Domain Multilingual Audio Deepfake benchmark).
  `benchmarks/XMAD/`, HF `SpeechAntiSpoofingBenchmarks/XMAD`.
- **Raw source:** `/home/kirill/mnt/users_4tb/datasets/xmad` — 7 langs × 2 corpora = 14 subsets, each with
  `real/`, `fake/`, `meta.csv` (`sample_name, is_fake, speaker_id, meta, split`).
- **Protocol → schema:** label ← `is_fake` (0=bonafide, 1=spoof). `path` = `<lang>/<corpus>/<real|fake>/<sample_name>`.
  `utterance_id` = `<lang>__<corpus>__<real|fake>__<stem>` (sample_name repeats across real/fake & corpora →
  full prefix needed; cf. [[reference_multisubset_uid_collision]]). `notes`: `language`, `corpus`, `speaker_id`, `split`.
  - **Expected counts: total 414858 / bonafide 207429 / spoof 207429** (50/50; per-subset breakdown in execution log).
- **License:** CC BY-NC-SA 4.0 → parquet OK. LICENSE.txt = CC BY-NC-SA 4.0.
- **arxiv:** `2506.00462`.
- **Build:** 16 kHz mono WAV PCM_16 → **CLEAN raw-byte embed**. ~70–80 GB → **~220 shards**.
  **Proxy OFF for push.**

## Manifest placement
- All three buildable datasets → **Core** (default). Core re-computes coverage for every existing
  submission — flagged in each manifest PR.

## 🚦 PLAN GATE — present the above; await explicit OK. Probe/build/push nothing before this.

---

# Execution log (filled autonomously after approval)

## DFADD  ✅ LIVE
- [x] Scaffold + clean-embed builder; whole-set probe 0/3755 fails → CLEAN
- [x] Build: 2 shards, 516 MB, counts asserted (755/3000); labels.parquet emitted
- [x] validate-dataset → D1–D7 green
- [x] Pushed `SpeechAntiSpoofingBenchmarks/DFADD` @ `c578c836da3b522b27d3dd85f89309f1737e5d31`; online sanity OK (3755 {755/3000})
- [x] Manifest PR #19 (core_set, n_trials 3755, paper_url arXiv 2409.08731) — MERGED (additive, 17 entries)
- [x] Baseline: EER 48.21%, 0 skipped; scores @ random-baseline-asas `78d28e1b…`; submission PR #1
- [x] Reproduced locally (sha matched, Δ EER 0.0) → filled reproduction → merged PR #1
- [x] Re-ingested (281 rows, DFADD row present, 0 DFADD warnings) → cache.json to Space
- [x] LIVE: Space RUNNING, /healthz ok, random-baseline gold, rank #25/25

## PyAra  ✅ LIVE
- [x] Scaffold + clean-embed builder; whole-set probe 0/201778 fails → CLEAN
- [x] Build: 100 shards, 32 GB, counts asserted (73583/128195); labels.parquet emitted
- [x] validate-dataset → D1–D7 green
- [x] Pushed `SpeechAntiSpoofingBenchmarks/PyAra` @ `40167badd6d8a289f02054761ee5411f49227b87` (proxy-off upload_large_folder); online sanity OK
- [x] Manifest PR #20 (core_set, n_trials 201778, paper_url kaggle) — MERGED (additive, 18 entries)
- [x] Baseline: EER 49.86%, 0 skipped; scores @ random-baseline-asas `2a2743fc…`; submission PR #1
- [x] Reproduced locally (sha matched, Δ EER 0.0) → submission PR #1 merged
- [x] Re-ingested (282 rows, PyAra row present, 0 PyAra warnings) → cache.json to Space
- [x] LIVE: Space RUNNING, /healthz ok

## XMAD  ✅ LIVE
- [x] Scaffold + clean-embed builder. **Source defect found:** 5 M-AILABS/MASC subsets ship only
      spoof audio (46,773 real files absent). Per user: keep them **spoof-only**, drop missing-real rows.
- [x] **Decode-verify folded into the shard build** (one disk pass, resumable per-shard) — the standalone
      368k-file probe on the contended HDD was the bottleneck and got reaped at ~60 min. 0 failures.
- [x] Build: 200 shards, 58 GB, counts asserted (dropped 46773 → 160656/207429 = 368085); labels.parquet
- [x] validate-dataset → D1–D7 green (368085 unique ids/paths)
- [x] Pushed `SpeechAntiSpoofingBenchmarks/XMAD` @ `fc1412809b875e1c5c102a2f615f30ffd09eda3d` (proxy-off, self-resuming); online sanity OK
- [x] Manifest PR #21 (core_set, n_trials 368085, paper_url arXiv 2506.00462) — MERGED (additive, 19 entries)
- [x] Baseline: EER 49.80%, 0 skipped; scores @ random-baseline-asas `27fff113…`; submission PR #1
- [x] Reproduced locally (sha matched, Δ EER 0.0) → submission PR #1 merged
- [x] Re-ingested (283 rows, XMAD row present, 0 XMAD warnings) → cache.json to Space
- [x] LIVE: Space RUNNING, /healthz ok, random-baseline gold

## Decisions (gate answered 2026-06-10)
- **TIMIT-TTS: SKIPPED** (spoof-only, no EER).
- Name confirmed **DFADD**.
- PyAra source/paper_url = **https://www.kaggle.com/datasets/alep079/pyara** (arxiv key empty; Kaggle in citation/paper_url).
- Proceed: DFADD, PyAra, XMAD → **Core**, seeded + landed autonomously.
