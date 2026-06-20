# Arena submission plan: AASIST

- **Source repo:** https://github.com/clovaai/aasist
- **Model name / slug:** AASIST / aasist  (slug lowercase)
- **Variant:** AASIST only (NOT AASIST-L)
- **Checkpoint:** `benchmarks/AASIST/AASIST.pth` (from clovaai/aasist
  `models/weights/AASIST.pth`, sha256 51d2d9cf…1a1c0, 1.2 MB) → to be mirrored to
  `SpeechAntiSpoofingBenchmarks/AASIST` and pinned at publish
- **Paper:** arXiv 2110.01200 — Jung et al., AASIST, ICASSP 2022 (ranked tiers)
- **params_millions:** 0.297866
- **Date:** 2026-06-02

## 0. Canon read
- [x] Read `docs/submitting/submit-model.md`, `docs/developing/new-model.md`,
      `docs/developing/testing-and-pitfalls.md`
- [x] Studied reference model: `benchmarks/ResCapsGuard/`

## 1. Wrapper (`benchmarks/AASIST/`)
- [x] `_net.py` — clovaai/aasist `models/AASIST.py` (self-contained, torch only)
- [x] `aasist.py` — `AntiSpoofingModel` subclass: `load()`, `score_batch()`, `unload()`
  - [x] higher score = more bona fide (`logits[:, 1]`; confirmed vs original eval)
  - [x] weights loaded in `load()`, not `__init__`
  - [x] no resampling (audio at `expected_sample_rate`; deterministic 64600 window)
- [x] `meta.yaml` — name, slug, description, code, checkpoint, paper, params
- [x] `sweep.py` — adapted from reference
- [x] `test_aasist.py` — load / score_batch / unload, batch>1 + batch=1 (5 passed)

## 2. Datasets (dynamic discovery, local-only)
- [x] Discovered = `benchmarks/*/` with `eval.yaml`: ASVspoof2019_LA,
      ASVspoof2021_DF, ASVspoof2021_LA, CD-ADD, InTheWild (5)
- [x] Registered each via `local set`; `local list`/`show` resolve cleanly
- [x] Never pass `--no-local`

## 3. Batch size (single RTX 4070 Ti SUPER, PCI index 1)
- [x] `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`; ordinal 0 = 4070 Ti SUPER

  | batch_size | utt/s |
  |:----------:|:-----:|
  | 1  | 76.9  |
  | 2  | 145.8 |
  | 4  | 192.0 |
  | 8  | 193.2 |
  | 16 | 194.2 |
  | 32 | 194.8 |  ← chosen (peak)
  | 64 | OOM   |

- [x] Chosen `batch_size = 32` (set in wrapper, dated comment)

## 4. Smoke validation (smallest dataset = InTheWild)
- [x] Ran on InTheWild; EER = 43.0146%, n_skipped = 0, n_trials = 31779
- [x] Reproduce self-consistency: recomputed EER == claimed (Δ 0.000e+00), full
      coverage vs pinned revision's local labels. (Canonical `reproduce
      --scoring` needs the scores uploaded to HF first; it runs identically in
      CI post-PR. Pre-upload, recomputed the metric via the package's own
      `reproduce` internals against local labels.)

## 🚦 GATE 1 — present plan + wrapper + batch size + smoke result; await OK

## 5. Full run (all discovered datasets, local)
- [x] Ran each dataset; `results/<DS>/` written; full coverage (scores == n_trials), 0 skips

  | dataset | EER% | n_trials | n_skipped |
  |---------|------|----------|-----------|
  | ASVspoof2019_LA (in-domain) | 0.829 | 71,237 | 0 |
  | ASVspoof2021_LA | 12.346 | 181,566 | 0 |
  | ASVspoof2021_DF | 17.040 | 611,829 | 0 |
  | InTheWild | 43.015 | 31,779 | 0 |
  | CD-ADD | 51.052 | 20,786 | 0 |

## 🚦 GATE 2 — present all results; on OK, run `submit` per dataset  [APPROVED]

## 6. Publish + PRs (manual path — `submit` would re-run the runner)
- [x] Created `SpeechAntiSpoofingBenchmarks/AASIST` (model repo, public);
      uploaded checkpoint + 5 `scores.txt` in one commit
      (sha `e842653505c2832ac9f46bbf56173b0f54ef82a7`)
- [x] Wrote `submissions/<DS>/aasist.yaml` (pinned scores_url + sha256, dataset
      revision from manifest), validated all 5 (OK)
- [x] `reproduce --scoring --no-local` on all 5 (CI-equivalent): sha matched,
      EER Δ 0.0e+00 each
- [x] Filled `reproduction` block (scoring), re-validated
- [x] Opened 5 HF PRs; CI `verify-pr` ✅ on all (schema/sha256/EER)
- [x] Merged all 5 (PRs #17/#4/#5/#6/#4); submissions live on `main`
- [x] Substituted pinned HF checkpoint URL into `meta.yaml`

## 7. Badges (after merge)
- [x] Per-dataset EER + dynamic tier/rank badges in `benchmarks/AASIST/README.md`,
      uploaded as the HF model card
- [x] Arena ingested: **tier = gold, rank = #1 of 4**

## Outcome
AASIST is live on the Arena (gold tier, #1 of 4). Full core coverage + paper.

## Notes / guideline discrepancies
- None found so far. (Record any here; propose fix + ask before editing official docs.)
