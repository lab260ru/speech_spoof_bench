# Arena submission plan: Whisper-MFCC-MesoNet (paper-faithful)

> Supersedes the base `whisper-mesonet` plan. After the base Whisper-only
> variant was found not to match the paper, the user asked to (a) use the
> fine-tuned Whisper+MFCC variant and (b) add the sox silence-trim. This plan
> tracks that paper-faithful submission.

- **Source repo:** <https://github.com/piotrkawa/deepfake-whisper-features>
- **Model name / slug:** Whisper-MFCC-MesoNet / `whisper-mfcc-mesonet`
- **Variant:** `whisper_frontend_mesonet`, `frontend_algorithm=[mfcc]`,
  `input_channels=2`, fine-tuned (`freeze_encoder=false`) — the paper's best
  MesoNet config.
- **Checkpoint:** `benchmarks/WhisperMFCCMesoNet/whisper_mfcc_mesonet_finetuned.pth`
  (Drive `all_models/whisper_mfcc_mesonet_finetuned/weights.pth`,
  sha256 `a34a00d0…ceee12`) → HF `SpeechAntiSpoofingBenchmarks/WhisperMFCCMesoNet`.
- **Paper:** arXiv:2306.01428 (INTERSPEECH 2023). ITW EER 26.72% for this config.
- **params_millions:** 7.660881
- **Date:** 2026-06-04
- **Compute:** free local GPU (RTX 5060 Ti, index 2; the 4070 Ti SUPERs were busy).

## 1. Wrapper (`benchmarks/WhisperMFCCMesoNet/`)
- [x] `_net.py` — Whisper encoder + MFCC front-end + `WhisperMultiFrontMesoNet`
- [x] `whispermfccmesonet.py` — `load`/`score_batch`/`unload`
  - [x] higher score = more bona fide (raw logit)
  - [x] weights in `load()`; no resampling
  - [x] sox silence-trim + 30 s repeat-pad (faithful upstream pipeline)
- [x] `meta.yaml`, `sweep.py`, `test_*.py` (4 passed), `implementation-notes.md`

## 2. Datasets — already registered (local list), 5 total. Never `--no-local`.

## 3. Batch size = 4 (free 16 GB card; base sweep peak, MFCC compute similar).

## 4. Runtime requirement — **libsox**
sox absent from env. Installed to `/tmp/soxenv`; every run/submit must set
`LD_LIBRARY_PATH=/tmp/soxenv/lib:$LD_LIBRARY_PATH` or `score()` aborts on clip 1.

## 5. Reproduction gate
- [x] In-the-Wild EER = **26.718%** vs paper 26.72% (Δ 0.01 pp), n_skipped=0,
      self-consistency recompute Δ=0.0.

## 6. Full run (all 5 datasets, local) — DONE
- [x] ASVspoof2021_DF 0.46 · ASVspoof2019_LA 5.83 · ASVspoof2021_LA 15.96 ·
      CD-ADD 18.90 · InTheWild 26.72 (all n_skipped=0, self-consistent Δ=0.0)

## 🚦 GATE 2 — approved.

## 7. Publish + PRs — DONE
- [x] Uploaded checkpoint + 5 scores.txt to HF WhisperMFCCMesoNet
      (pinned sha 71dff409…)
- [x] 5 submission YAMLs written + schema-validated +
      `reproduce --scoring --no-local` matched against pinned revisions (Δ=0.0)
- [x] 5 HF PRs opened; `verify-pr` ✅ on all (3 needed one-at-a-time
      re-dispatch due to HF concurrency burst-drop)
- [x] reproduction block filled (match: scoring) + all 5 PRs merged to main

## 8. Badges — DONE
- [x] README/model card uploaded with per-dataset EER + dynamic tier/rank badges
- [x] result.yaml projections uploaded to model repo `.eval_results/`
- [x] Live Arena: **tier gold, rank #1 of 7**

## Outcome
Reproduced the paper's In-the-Wild EER (26.72%). Model live on the Arena at
`?system=whisper-mfcc-mesonet`.

## Notes
- Base Whisper-only `WhisperMesoNet` dir kept for reference (not submitted).
- Env gap: `libsox.so` not shipped with torchaudio 2.7; worked around via
  `/tmp/soxenv`. Consider vendoring libsox or adding `sox` to the project env.
