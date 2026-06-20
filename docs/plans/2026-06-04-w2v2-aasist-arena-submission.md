# Arena submission plan: W2V2-AASIST

- **Source repo:** <https://github.com/TakHemlata/SSL_Anti-spoofing>
- **Model name / slug:** W2V2-AASIST / w2v2-aasist
- **Checkpoint:** `benchmarks/W2V2-AASIST/LA_model.pth` (LA variant, from the
  repo's Google-Drive Pre_trained_models folder) + build-dep `xlsr2_300m.pt`
  (base XLS-R 300M, fairseq). HF mirror URL filled at publish.
- **Paper:** arXiv 2202.12233 — Tak et al., "Automatic speaker verification
  spoofing and deepfake detection using wav2vec 2.0 and data augmentation",
  Odyssey 2022.
- **params_millions:** 317.8378 (counted from the loaded net)
- **Date:** 2026-06-04

## 0. Canon read
- [x] Read `docs/submitting/submit-model.md`, `docs/developing/new-model.md`,
      `docs/developing/testing-and-pitfalls.md`
- [x] Studied reference models: `benchmarks/AASIST/` (same AASIST back-end),
      `benchmarks/RawTFNet/` (most recent submission)

## 1. Wrapper (`benchmarks/W2V2-AASIST/`)
- [x] `_net.py` — vendored from source `model.py` (SSLModel = fairseq wav2vec2
      XLS-R 300M front-end + AASIST back-end). Two minimal edits: file-relative
      `xlsr2_300m.pt` path; tolerant cross-version wav2vec2 build (see notes).
- [x] `w2v2_aasist.py` — `AntiSpoofingModel` subclass: `load()`, `score_batch()`,
      `unload()`
  - [x] higher score = more bona fide (`logits[:, 1]`, confirmed vs source
        `batch_score = batch_out[:, 1]`)
  - [x] weights loaded in `load()`, not `__init__`
  - [x] no resampling (audio is at `expected_sample_rate = 16000`)
  - [x] `.to(device)` BEFORE `.eval()` so SSL submodule stays in eval mode
- [x] `meta.yaml` — name, slug, description, code, checkpoint, paper, params
- [x] `sweep.py` — adapted from AASIST/RawTFNet
- [x] `test_w2v2_aasist.py` — pad / strict-load / score_batch (incl. batch=1) /
      determinism — **5 passed**

## 2. Datasets (dynamic discovery, local-only)
- [x] Discovered datasets = `benchmarks/*/` dirs containing `eval.yaml`:
      ASVspoof2019_LA, ASVspoof2021_DF, ASVspoof2021_LA, CD-ADD, InTheWild (5)
- [x] Already registered in the local registry (`local list` confirms all 5)
- [x] Never pass `--no-local`

## 3. Batch size (single RTX 4070 Ti Super)
- [x] `CUDA_DEVICE_ORDER=PCI_BUS_ID; CUDA_VISIBLE_DEVICES=3` (index 1 busy ~1.1 GB;
      torch ordinal 0 confirmed as the 4070 Ti SUPER)
- [x] Ran `sweep.py`; throughput table:

  | batch_size | utt/s |
  |:----------:|:-----:|
  | 1  | 28.0  |
  | 2  | 50.4  |
  | 4  | 84.1  |
  | 8  | 115.6 |
  | 16 | 121.8 |
  | 24 | 123.2 |  ← chosen (peak)
  | 32 | 121.8 |

- [x] Chosen `batch_size = 24` (peak; 16/24/32 plateau within ~1.5%)

## 4. Smoke validation (smallest discovered dataset = CD-ADD, 20,786 trials)
- [x] EER = 38.569%, n_skipped = 0, full coverage (20786/20786)
- [x] Independent EER recompute from our own `scores.txt` + local labels
      (mirrors CI `reproduce` internals) matches `result.yaml` exactly
      (Δ 0.0e+00). Full HF-URL `reproduce --scoring` runs post-upload.

## 🚦 GATE 1 — present plan + wrapper + batch size + smoke result; await OK

## 5. Full run (all discovered datasets, local)
- [x] Ran each dataset; `results/<DS>/` written. All self-consistency Δ 0.0e+00,
      0 skips, full coverage.

  | dataset | EER% | n_trials | n_skipped | AASIST (same bench) |
  |---------|------|----------|-----------|---------------------|
  | ASVspoof2019_LA | 0.224 | 71,237 | 0 | 0.829 |
  | ASVspoof2021_LA | 8.113 | 181,566 | 0 | 12.346 |
  | ASVspoof2021_DF | 8.318 | 611,829 | 0 | 17.040 |
  | InTheWild | 11.222 | 31,779 | 0 | 43.015 |
  | CD-ADD | 38.569 | 20,786 | 0 | 51.052 |

## 🚦 GATE 2 — present all results; on OK, run `submit` per dataset

## 6. Publish + PRs (manual two-upload path, per new-model.md)
- [x] Uploaded `LA_model.pth` + 5 `scores.txt` to
      `SpeechAntiSpoofingBenchmarks/W2V2-AASIST` @ `75ec0a4aa6491cee7748c12996a87561d1b02fb7`
- [x] One submission yaml + HF PR per dataset:
      ASVspoof2019_LA #22, ASVspoof2021_LA #9, ASVspoof2021_DF #9,
      InTheWild #9, CD-ADD #11
- [x] CI (`verify-pr`) **green on all 5** (schema/sha256/EER ✅). First burst hit
      the documented concurrency-cap (1 pass / 3 cancelled / 1 fail); re-dispatched
      the 4 serially via PR-ref commits → all PASS. Not a reproduce failure.
- [ ] **Merge** (maintainer) + fill `reproduction` block (empty = Arena hides it)

- [x] **Merged all 5** + reproduction blocks filled (`match: scoring`) on main.
- [x] **Arena: gold tier, rank #1 of 8.**

## 7. Badges (after each merge)
- [x] Per-dataset EER + dynamic tier/rank badges in `benchmarks/W2V2-AASIST/README.md`
      and pushed as the HF model card; 5 backlink `result.yaml` files added to the
      model repo. Badge endpoint live (tier=gold, rank=#1 of 8).

## Notes / guideline discrepancies
- fairseq cross-version load: `load_model_ensemble_and_task` fails on the
  xlsr2_300m.pt cfg (`multiple_train_files` not in this fairseq's task config).
  Worked around by building wav2vec2 directly from the model cfg via
  `merge_with_parent(Wav2Vec2Config(), cfg.model, remove_missing=True)`. This is
  a wrapper-internal fix, not an official-doc change. Recorded in
  `implementation-notes.md`.
- This is the first >1M-param / SSL model: needs fairseq (installed) + the 3.8 GB
  base `xlsr2_300m.pt` to construct the architecture. Build dependency documented.
