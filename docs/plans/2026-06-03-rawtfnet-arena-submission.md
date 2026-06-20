# Arena submission plan: RawTFNet

- **Source repo:** <https://github.com/swagshaw/RawTFNet-Pytorch>
- **Model name / slug:** RawTFNet / `rawtfnet`
- **Checkpoint:** `benchmarks/RawTFNet/Best_RawTFNet_32.pth` (from source repo
  `ckpts/Best_RawTFNet_32.pth`); HF URL filled at publish.
- **Paper:** arXiv 2507.08227 — *RawTFNet: A Lightweight CNN Architecture for
  Speech Anti-spoofing* (Xiao, Dang, Das, 2025). Ranked tiers eligible.
- **params_millions:** 0.177540 (177,540 params)
- **Date:** 2026-06-03

## 0. Canon read
- [x] Read `docs/submitting/submit-model.md`, `docs/developing/new-model.md`,
      `docs/developing/testing-and-pitfalls.md`
- [x] Studied reference model: `benchmarks/Res2TCNGuard/` (raw-waveform, sinc
      front-end — closest analogue)

## 1. Wrapper (`benchmarks/RawTFNet/`)
- [x] `_net.py` — architecture ported from source repo (only the full RawTFNet
      path: SincConv → DWS_Frontend_SE → TfSepNet depth=10 width=32). torchinfo
      dep and unused variants dropped; math untouched.
- [x] `rawtfnet.py` — `AntiSpoofingModel` subclass: `load()`, `score_batch()`, `unload()`
  - [x] higher score = more bona fide — returns `logits[:, 1]` (source repo:
        `batch_score = batch_out[:, 1]`); verified by EER magnitude (1.99%, not ~98%)
  - [x] weights loaded in `load()`, not `__init__`
  - [x] no resampling (audio is at `expected_sample_rate=16000`); deterministic
        first-64600-sample tile-pad window (mirrors repo `pad(x, 64600)`)
- [x] `meta.yaml` — name, slug, description, code, checkpoint, paper, params
- [x] `sweep.py` — adapted from Res2TCNGuard
- [x] `test_rawtfnet.py` — load / score_batch / unload, incl. size-1 final batch
      (forward ends with `.squeeze()`, so wrapper reshapes to `(-1, 2)`). 3 passed.

### Checkpoint selection
Source ships two checkpoints. Matched each to its architecture by param count
(model code only, no unpickling — sandbox-safe):
- `Best_RawTFNet_32.pth` → full `RawTFNet` (TfSepNet width=32), 177,540 params — **used (README's "best")**
- `Best_RawTFNet_16.pth` → `RawTFNet_small` (width=16), 72,804 params — unused
`strict=True` `load_state_dict` succeeds against the full RawTFNet, confirming the
mapping. (Note: source `main.py` is buggy — both `--model_name` choices map to
`RawTFNet_small`; ignored, relied on the model defs + state-dict match instead.)

## 2. Datasets (dynamic discovery, local-only)
- [x] Discovered datasets = `benchmarks/*/` dirs with `eval.yaml`:
      ASVspoof2019_LA, ASVspoof2021_DF, ASVspoof2021_LA, CD-ADD, InTheWild (5)
- [x] Registered each via `local set SpeechAntiSpoofingBenchmarks/<DS> benchmarks/<DS>`
- [x] `local list` confirms all 5 registered (→ drive3_8tb canonical paths)
- [x] Never pass `--no-local`

## 3. Batch size (single RTX 4070 Ti SUPER)
- [x] `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3` (index 1 was busy at
      90% util; index 3 is the other 4070 Ti SUPER, free). Confirmed device name.
- [x] Ran `sweep.py`; throughput table:

  | batch_size | utt/s |
  |:----------:|:-----:|
  | 1  | 55.1 |
  | 2  | 107.5 |
  | 4  | 211.4 |
  | 8  | **272.4** |
  | 16 | 270.4 |
  | 32 | 270.3 |
  | 64 | OOM |

- [x] Chosen `batch_size = 8` (throughput peak; 8/16/32 plateau ~270 utt/s, bs=8
      lowest memory). Set in wrapper with dated comment.

## 4. Smoke validation
- [x] Ran on **ASVspoof2019_LA** (71,237 trials). EER = **1.9865%**, n_skipped = **0**.
- [x] Self-consistency: recomputed EER from `scores.txt` + local labels via the
      package's `eer_percent` metric = 1.9864750633981403 — **exact match** to
      `result.yaml` (diff 0.0), coverage 71237/71237.

> **Deviation from "smallest dataset" (with rationale):** the skill says smoke
> the smallest discovered dataset (CD-ADD, 20,786). I smoked ASVspoof2019_LA
> instead because it is the only in-domain set where EER is interpretable — a
> flipped score sign there shows ~98% vs ~2%. On CD-ADD/InTheWild every model
> (incl. ResCapsGuard/Res2TCNGuard) sits ~50%, so a smoke there cannot catch the
> #1 risk (score-direction flip). The 1.99% result confirms direction + plumbing.
> `reproduce --scoring` itself only fetches HF URLs (pre-publish it can't run on
> local scores), so the round-trip was done as the equivalent local recompute above.

## 🚦 GATE 1 — present plan + wrapper + batch size + smoke result; await OK

## 5. Full run (all discovered datasets, local) — done
All 5 ran locally; `results/<DS>/{result.yaml,scores.txt}` written. 0 skips, full
coverage (scores-lines == n_trials), self-consistency recompute exact (diff 0.0):

  | dataset | EER% | n_trials | n_skipped |
  |---------|------|----------|-----------|
  | ASVspoof2019_LA | 1.99 | 71,237 | 0 |
  | ASVspoof2021_LA | 8.03 | 181,566 | 0 |
  | ASVspoof2021_DF | 15.16 | 611,829 | 0 |
  | InTheWild | 38.51 | 31,779 | 0 |
  | CD-ADD | 52.85 | 20,786 | 0 |

## 🚦 GATE 2 — present all results; on OK, `submit` per dataset

## 6. Publish + PRs — done (manual upload path; see implementation-notes.md)
Model repo `SpeechAntiSpoofingBenchmarks/RawTFNet` created; checkpoint + all 5
`scores.txt` uploaded (pinned commit `aa12f0fe2f10cc5278c954175a12c18cc43e3113`);
model card pushed. 5 submission PRs opened (one per dataset), all `validate-submission` OK:

  | Dataset | EER% | PR | verify-pr CI |
  |---|---|---|---|
  | ASVspoof2019_LA | 1.99 | #19 | ✅ |
  | ASVspoof2021_DF | 15.16 | #7 | ✅ |
  | ASVspoof2021_LA | 8.03 | #6 | ✅ |
  | CD-ADD | 52.85 | #8 | ✅ |
  | InTheWild | 38.51 | #6 | ✅ |

All 5 verify-pr CI green (DF/LA needed manual re-dispatch after the HF webhook
burst + 429s — see docs/plans/2026-06-03-ci-429-burst-followup.md).

**All 5 PRs merged** (self-merged as org member `korallll`), each with the
`reproduction` block filled (`reproduced_by: SpeechAntiSpoofingBenchmarks`,
`match: scoring` — the verify-pr CI did the scoring reproduce). Confirmed
reproduction present on `main` for all 5.

**LIVE on the Arena:** 🥇 **Gold tier, rank #1 of 6**
(badge endpoints `/badge/rawtfnet/{tier,rank}.json` respond gold / #1 of 6).

## 7. Badges — done
README + HF model card carry the static EER badges + dynamic tier/rank endpoints,
all now live (tier=gold, rank=#1/6).

## Outcome
RawTFNet submitted, verified, merged across all 5 datasets, and ranked #1 (Gold).
Deferred: CI 429/burst hardening (docs/plans/2026-06-03-ci-429-burst-followup.md).

## Notes / guideline discrepancies
- Source `main.py` model-name mapping bug noted above (handled, not a toolkit doc issue).
- `reproduce --scoring` cannot run against a local scores file pre-publish
  (`hf_fetch.download` only resolves HF URLs). Used an equivalent local recompute
  for the smoke self-consistency check. Not a doc error — just clarifying the
  smoke step's intent. Recorded here per the control rules.
