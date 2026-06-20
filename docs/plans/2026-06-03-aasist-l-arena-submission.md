# Arena submission plan: AASIST-L

- **Source repo:** https://github.com/clovaai/aasist
- **Model name / slug:** AASIST-L / aasist-l  (slug lowercase)
- **Checkpoint:** `benchmarks/AASIST-L/AASIST-L.pth` (426 KB, sha256
  814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a);
  HF mirror `SpeechAntiSpoofingBenchmarks/AASIST-L` (pinned at publish)
- **Paper:** arxiv 2110.01200 — AASIST (ICASSP 2022); AASIST-L is the lightweight
  config from the same paper/repo → ranked tiers
- **params_millions:** 0.085306 (85,306 params, counted from loaded net)
- **Date:** 2026-06-03

## 0. Canon read
- [x] Read submit-model.md, new-model.md, testing-and-pitfalls.md
- [x] Studied reference model: `benchmarks/AASIST/` (same architecture family)

## 1. Wrapper (`benchmarks/AASIST-L/`)
- [x] `_net.py` — copied verbatim from AASIST (shared `Model` class)
- [x] `aasist_l.py` — `AASIST_L(AntiSpoofingModel)`: `load()`, `score_batch()`, `unload()`
  - [x] higher score = more bona fide (`logits[:, 1]`)
  - [x] weights loaded in `load()`, not `__init__`
  - [x] no resampling (audio is at `expected_sample_rate = 16000`)
- [x] `meta.yaml` — name, slug, description, code, checkpoint, paper, params
- [x] `sweep.py` — adapted from AASIST
- [x] `test_aasist_l.py` — load / score_batch / unload, batch=1 path; 5 passed
- [x] AASIST-L config: filts […,[32,24],[24,24]], gat_dims [24,32],
      pool_ratios [0.4,0.5,0.7,0.5]; strict load OK

## 2. Datasets (dynamic discovery, local-only)
- [x] Discovered (`benchmarks/*/eval.yaml`): ASVspoof2019_LA, ASVspoof2021_DF,
      ASVspoof2021_LA, CD-ADD, InTheWild (5)
- [x] Registered each via `local set`; `local list` confirms all 5
- [x] Never pass `--no-local`

## 3. Batch size (single RTX 4070 Ti SUPER)
- [x] `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3` (index 1 busy; index 3
      is a free 4070 Ti SUPER)
- [x] Ran `sweep.py`; peak 256.5 utt/s at bs=16 (1:75, 2:144, 4:247, 8:252,
      16:256, 32:251, 64:OOM)
- [x] Chosen `batch_size = 16` (set in wrapper with dated comment)

## 4. Smoke validation (smallest discovered dataset = CD-ADD, 20,786 trials)
- [x] Ran on CD-ADD; EER = 50.715%, n_skipped = 0 (matches AASIST's ~51% on this
      hard OOD set — not a sign flip; direction byte-identical to AASIST)
- [x] Self-consistency recompute via reproduce's internal path (local scores +
      local labels + eer_percent fn): delta = 0.0

## 🚦 GATE 1 — present plan + wrapper + batch size + smoke result; await OK

## 5. Full run (all discovered datasets, local)
- [x] Ran each on free RTX 5060 Ti (index 2); `results/<DS>/` written; all 0 skipped

  | dataset | EER% | n_trials | n_skipped |
  |---------|------|----------|-----------|
  | ASVspoof2019_LA | 0.992 | 71,237 | 0 |
  | ASVspoof2021_LA | 13.153 | 181,566 | 0 |
  | ASVspoof2021_DF | 15.959 | 611,829 | 0 |
  | InTheWild | 44.448 | 31,779 | 0 |
  | CD-ADD | 50.715 | 20,786 | 0 |

## 🚦 GATE 2 — present all results; on OK, submit per dataset

## 6. Publish + PRs (manual two-upload path)
- [x] Published model repo `SpeechAntiSpoofingBenchmarks/AASIST-L` @ sha
      `e4185b270ec20077c918e06a45093717a1bd5e30` (checkpoint + _net + wrapper +
      README + 5 scores.txt under `.eval_results/`)
- [x] All 5 `reproduce --scoring` passed (sha matched, Δ 0.0e+00)
- [x] One PR per dataset:
      - ASVspoof2019_LA → discussions/18
      - ASVspoof2021_LA → discussions/5
      - ASVspoof2021_DF → discussions/6
      - InTheWild → discussions/5
      - CD-ADD → discussions/7
- [x] CI (`verify-pr`) green on all 5 (✅ schema/sha256/EER match). #18 needed a
      re-dispatch (initial webhook miss — the documented "no retries" gotcha).
- [x] All 5 PRs merged; `reproduction` block filled on each (match: scoring,
      reproduced_by SpeechAntiSpoofingBenchmarks, 2026-06-03). Arena surfaces
      AASIST-L after the next ingest cycle (~30 min TTL).

## 7. Badges
- [x] Per-dataset EER + tier/rank badges added to `benchmarks/AASIST-L/README.md`
      and the HF model card (slug `aasist-l`). Endpoint (tier/rank) badges and
      `?system=aasist-l` links go live once the 5 PRs are merged and the Arena
      re-ingests.

## Notes / guideline discrepancies
- Skill said docs live at `speech-spoof-bench/docs/...` from the repo root; the
  actual path has an extra nesting level `speech-spoof-bench/speech-spoof-bench/docs/...`.
  Minor skill-doc path drift; recorded only, not a canonical-doc edit.
- AASIST and AASIST-L share `_net.py` exactly; only `model_config` + checkpoint differ.
