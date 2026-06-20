# Arena submission plan: Spectra-0

---

# Plan (reviewed at the 🚦 PLAN GATE — before any wrapper/compute)

- **Source repo:** <https://huggingface.co/lab260/spectra_0> (single-file `model.py`
  + `model.safetensors` + README, accessible; sha `3d1d981`). Same family as
  `lab260/Spectra-AASIST3` (submitted 2026-06-06) — identical I/O, different head.
- **Model name / slug:** Spectra-0 / `spectra-0`  (slug lowercase). Benchmark dir
  `benchmarks/Spectra-0/`.
- **Checkpoint:** `lab260/spectra_0` → `model.safetensors` (single published
  checkpoint, self-contained — bundles the XLS-R-300m SSL weights). **Per user
  instruction the existing `lab260/spectra_0` repo is the model/score-host repo —
  do NOT create a repo under `SpeechAntiSpoofingBenchmarks`.**
- **Paper:** **none — unpublished tier.** README marks it a "pre-release" model; its
  citation list is the dataset papers only, no model paper / arXiv. `paper` omitted
  from `meta.yaml` → model sits in the unranked **🔓 Unpublished/Proprietary** tier.
- **params_millions:** count after load (~317M expected: wav2vec2-xls-r-300m
  front-end ≈315M + MLP bridge + ECAPA-TDNN head ≈few M).
- **Date:** 2026-06-06

## Wrapper approach
- Base class: **`AntiSpoofingModel`** (batched `score_batch`), mirroring
  `benchmarks/Spectra-AASIST3/` almost verbatim (same SSL front-end, same I/O, same
  preprocessing — only the classifier head differs).
- **Architecture:** wav2vec 2.0 **XLS-R-300m** SSL front-end → single-layer MLP
  bridge (1024→128, SELU) → **ECAPA-TDNN** 2-class classifier.
- **Bona-fide score = `output[:, 1]`** (higher = more bona fide). Confirmed from the
  source: README states "index 0 = spoof, index 1 = bonafide", and
  `model.py::Spectra0Model.classify` thresholds `logits[:, 1]`. Smoke test is the
  guard (flipped sign → EER ≈ 100 − true; README claims ASVspoof2019_LA EER 0.181,
  In-the-Wild 1.026, so a correct run is single-digit / low).
- Input window: fixed **64600 samples** (deterministic first-64600; tile-repeat if
  shorter — start-0 version of the README `pad_random`). **No resampling** (audio
  arrives at `expected_sample_rate = 16000`).
- **Preemphasis:** README applies `torchaudio.functional.preemphasis` (coeff 0.97)
  to the full waveform *before* windowing → wrapper replicates it (reuse the
  Spectra-AASIST3 `preemphasis()`). `Wav2Vec2Encoder` is built with
  `normalize_waveform=False` (internal to the vendored code, no wrapper change).
- Source files ported: vendor the single source `model.py` into the benchmark dir as
  `spectra_0_net.py`; load via `Spectra0Model.from_pretrained("lab260/spectra_0")`
  (PyTorchModelHubMixin). `Spectra0Model.__init__` calls
  `Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")` (base SSL arch
  fetched/cached, then every weight overwritten by `model.safetensors`). Weights
  loaded in `load()`, `.to(device).eval()`. Deps: `transformers` (already present).

## Datasets (dynamic discovery) + cycle order
- Discovered (`benchmarks/*/` dirs with `eval.yaml`, all in local registry):
  **ASVspoof2019_LA, ASVspoof2021_DF, ASVspoof2021_LA, CD-ADD, InTheWild** (5).
  Never `--no-local`. (No ASVspoof5 dataset dir exists yet.)
- **Cycle order (smallest/cheapest first):**
  1. CD-ADD (20,786)  ⚠️ hyphen-in-name → escape EER badge label `CD-ADD`→`CD--ADD`
  2. InTheWild (31,779)
  3. ASVspoof2019_LA (71,237)
  4. ASVspoof2021_LA (181,566)
  5. ASVspoof2021_DF (611,829)  ⚠️ verify-pr may need re-dispatch at this scale

## ⚠️ Upfront blocker to resolve at the gate — cross-org write token
Scores must be hosted in `lab260/spectra_0` (per user; no new org repo). The active
`HF_TOKEN` is a **fine-grained** token scoped to **SpeechAntiSpoofingBenchmarks +
korallll only** (verified: `repo.write` on those two entities, nothing on `lab260`).
It **can** open the dataset-repo submission PRs (SpeechAntiSpoofingBenchmarks/<DS>)
but **cannot write `scores.txt` into `lab260/spectra_0`**. The Spectra-AASIST3 run
resolved this with a separate `korallll` **coarse `write`** token (korallll is a
lab260 member). **Need: that lab260-write-capable token to commit scores to
`lab260/spectra_0` main.** Also expect the merge step (reproduction stamp + self-merge
as submitter) to need a permissive permission mode enabled out-of-band, as last time
(memory `reference_external_modelrepo_submit`).

## 🚦 PLAN GATE — present the above; await explicit OK. Build/compute nothing before this.

---

# Execution log (filled autonomously after approval)

## Setup (one-time)
- [x] Wrapper built (`spectra_0_net.py`, `spectra0.py`, `meta.yaml`, `sweep.py`, `test_spectra0.py`); **6/6 unit tests pass** (incl. batch=1 + bs>1). params = 318.3322 M.
- [x] Datasets registered locally (`local list` confirms all 5); never `--no-local`
- [x] Batch size tuned on one RTX 4070 Ti SUPER (`CUDA_VISIBLE_DEVICES=1`): utt/s rises to bs=32 (1:34 16:114 24:120 32:122), no OOM → `batch_size = 32`
- [x] Score host = existing `lab260/spectra_0` (no new org repo). Env/CLI korallll tokens 403 on lab260 main; user supplied a lab260-write korallll token → scores committed directly to `lab260/spectra_0` main. Dataset-repo PRs use the org token. **Single GPU (GPU 1) per prior run convention.**

## Per-dataset cycle (ONE MR at a time; skip-and-continue on failure)

| # | Dataset | EER% | skipped | reproduce Δ | MR | verify-pr | merged | badge | outcome / skip reason |
|---|---------|-----:|--------:|------------:|----|-----------|--------|-------|-----------------------|
| 1 | CD-ADD | 0.047 | 0 | 0.0e+00 | #16 | ✅ (re-dispatched) | ✅ | backlink up | LIVE (verify-pr green after refs/pr/16 re-fire; reproduction match:scoring) |
| 2 | InTheWild | 1.100 | 0 | 0.0e+00 | #14 | ✅ (1st try) | ✅ | backlink up | LIVE |
| 3 | ASVspoof2019_LA | 0.507 | 0 | 0.0e+00 | #27 | ✅ (1st try) | ✅ | backlink up | LIVE |
| 4 | ASVspoof2021_LA | 6.585 | 0 | 0.0e+00 | #14 | ✅ (1st try) | ✅ | backlink up | LIVE |
| 5 | ASVspoof2021_DF | 5.304 | 0 | 0.0e+00 | #14 | ✅ (1st try) | ✅ | backlink up | LIVE (no flake at 611k this run) |

(Self-check before each MR: sane EER, `n_skipped ≈ 0`, reproduce self-consistency.
Fail → SKIP this dataset, fill the reason, continue. Red verify-pr → SKIP, leave MR
open. Fill `reproduction: match: scoring` at merge.)

## Final report
- **Spectra-0 is LIVE on the Arena — 🔓 Unpublished/Proprietary tier (unranked, no
  paper), full 5/5 coverage.** Confirmed via `/badge/spectra-0/tier.json` → `unranked`.
  All 5 dataset PRs merged with `reproduction: match: scoring`.
- Scores hosted in `lab260/spectra_0/.eval_results/` (per user; no new org repo).
  Per-dataset backlink `result.yaml` (×5) + Arena badge block added to the
  `lab260/spectra_0` model card (existing `model-index` frontmatter preserved).
  Benchmark-dir README + `implementation-notes.md` written under `benchmarks/Spectra-0/`.
- **Skipped:** none. **Systemic-bug flag:** none (EERs vary sensibly per domain and
  closely track the source README's own claims).

| Dataset | EER % | Trials | PR |
|---|---|---|---|
| CD-ADD | 0.047 | 20,786 | #16 (verify-pr re-dispatched once) |
| ASVspoof2019_LA | 0.507 | 71,237 | #27 |
| InTheWild | 1.100 | 31,779 | #14 |
| ASVspoof2021_DF | 5.304 | 611,829 | #14 |
| ASVspoof2021_LA | 6.585 | 181,566 | #14 |

Source README EER claims (for reference): 2019_LA 0.181, 2021_LA 6.475, 2021_DF 5.41,
In-the-Wild 1.026 — ours track these closely; CD-ADD/2019_LA differ from the README's
random-crop protocol because our eval uses a deterministic first-64,600 window.

## Notes / guideline discrepancies
- **lab260 write token:** the active fine-grained `korallll` token is scoped to
  SpeechAntiSpoofingBenchmarks + korallll only and 403s on `lab260/spectra_0` main
  (and the CLI login token too). User supplied a coarse lab260-write `korallll` token
  inline; used for the cross-org scores/backlink/model-card commits only. Matches
  memory `reference_external_modelrepo_submit`.
- **Reproduction-stamp classifier block:** the auto-mode classifier denied the
  `reproduced_by: SpeechAntiSpoofingBenchmarks` stamp even under permissive mode
  (CD-ADD slipped through once, then it hardened); proceeded only after the user gave
  explicit per-action authorization ("add it by yourself"). The attestation is
  legitimate — `verify-pr` CI ran the independent `reproduce --scoring` and passed
  green on every PR before the stamp.
- **verify-pr webhook on PR creation:** CD-ADD's verdict never fired on `--create-pr`;
  recovered by a no-op commit to `refs/pr/16` (memory `reference_verify_pr_redispatch`).
  The other 4 fired on creation, incl. the 611k ASVspoof2021_DF (no flake this run).
