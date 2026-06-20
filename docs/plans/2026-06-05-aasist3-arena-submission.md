# Arena submission plan: AASIST3

---

# Plan (reviewed at the 🚦 PLAN GATE — before any wrapper/compute)

- **Source repo:** <https://github.com/mtuciru/AASIST3> (HF model: <https://huggingface.co/lab260/AASIST3>)
- **Model name / slug:** AASIST3 / `aasist3`  (slug lowercase)
- **Checkpoint:** `lab260/AASIST3` → `model.safetensors` (1.29 GB, self-contained:
  includes the wav2vec2 XLS-R-53 SSL front-end weights). Single published
  checkpoint in the repo — no "best of N" choice. **Per user instruction, the
  existing `lab260/AASIST3` repo is the model repo — do NOT create a repo under
  the `SpeechAntiSpoofingBenchmarks` org.** Write access confirmed (`korallll` is
  a `lab260` member).
- **Paper:** AASIST3 (Borodin et al.), *ASVspoof 2024 Workshop* —
  **arXiv 2408.17352** (DOI 10.21437/ASVspoof.2024-8).
- **params_millions:** 321.7495 (counted from `model.safetensors`)
- **Date:** 2026-06-05

## Wrapper approach
- Base class: **`AntiSpoofingModel`** (batched `score_batch`), mirroring
  `benchmarks/W2V2-AASIST/` (same XLS-R + AASIST family).
- **Bona-fide score = `output[:, 1]`** (logit of class 1; higher = more bona fide).
  - Confirmed from the **source training + eval code**, NOT the README quick-start.
    `datasets/generic.py` label map = `0 if "spoof" else 1` (bonafide = **1**);
    `utils/validation.py::compute_scores` takes `outputs[:, 1]` as the score and
    `compute_antispoofing_metrics` treats `label == 1` as genuine. The HF README
    "Quick Start" snippet claiming `0 = bonafide` is generic boilerplate that
    contradicts the actual pipeline — **ignored**. The smoke test is the guard:
    a flipped sign shows up as EER ≈ (100 − true).
- Input window: fixed **64600 samples** (~4 s @ 16 kHz; the model's `pos_S`/`pos_T`
  positional embeddings are sized for exactly this length). Deterministic eval
  window = first 64600 samples; tile-repeat if shorter (start-0 version of the
  source `apply_random_segment_extraction`). **No resampling** (audio arrives at
  `expected_sample_rate = 16000`).
- **Preemphasis:** the source eval pipeline applies
  `torchaudio.functional.preemphasis` (coeff 0.97) to the full waveform *before*
  windowing — the model was trained on preemphasized audio — so the wrapper
  replicates it. (The SSL encoder additionally normalizes the waveform by
  `÷ max|x|` internally.) The smoke EER confirms this preprocessing is right.
- Source files ported: vendor the source `model/` package into the benchmark dir
  as `aasist3_net/` (full_model, wav2vec, kan, gat, hs_gal, branch, pool,
  residual; pure-torch KAN, no extra deps). Load via
  `aasist3.from_pretrained("lab260/AASIST3")` (config sets `load_pretrained=false`,
  so the XLS-R base is built from the HF `wav2vec2-large-xlsr-53` config and then
  every weight is overwritten by `model.safetensors` — no separate base checkpoint
  download, unlike W2V2-AASIST). Weights loaded in `load()`, `.to(device)` then
  `.eval()`. New deps vs W2V2-AASIST: `transformers` (Wav2Vec2Model) — verify in env.

## Datasets (dynamic discovery) + cycle order
- Discovered (`benchmarks/*/` dirs with `eval.yaml`): **ASVspoof2019_LA,
  ASVspoof2021_DF, ASVspoof2021_LA, CD-ADD, InTheWild** (5). All already in the
  local registry (`local list`). Never `--no-local`.
- **Cycle order (smallest/cheapest first):**
  1. CD-ADD (20,786)
  2. InTheWild (31,779)
  3. ASVspoof2019_LA (71,237)
  4. ASVspoof2021_LA (181,566)
  5. ASVspoof2021_DF (611,829)

## 🚦 PLAN GATE — present the above; await explicit OK. Build/compute nothing before this.

---

# Execution log (filled autonomously after approval)

## Setup (one-time)
- [x] Wrapper built (`aasist3_net/`, `aasist3.py`, `meta.yaml`, `sweep.py`, `test_aasist3.py`); **6/6 unit tests pass** (incl. batch=1 + bs>1). Loads on transformers 5.3.0.
- [x] Datasets registered locally (`local list` confirms all 5); never `--no-local`
- [x] Batch size tuned on one RTX 4070 Ti Super (`CUDA_VISIBLE_DEVICES=1`); throughput plateaus ~38-39 utt/s for bs≥8 → `batch_size = 16`
- [x] Model repo = **existing `lab260/AASIST3`** (no new org repo). **Access note:** korallll fine-grained token is scoped to SpeechAntiSpoofingBenchmarks+korallll only (no lab260) → user supplied a lab260-scoped token; scores committed **directly to lab260/AASIST3 main** with it. Submission PRs on dataset repos use the original org-scoped token.

## Per-dataset cycle (ONE MR at a time; skip-and-continue on failure)

| # | Dataset | EER% | skipped | reproduce Δ | MR | verify-pr | merged | badge | outcome / skip reason |
|---|---------|-----:|--------:|------------:|----|-----------|--------|-------|-----------------------|
| 1 | CD-ADD | 30.727 | 0 | 0.0e+00 | #14 | ✅ | ✅ | live | LIVE (bronze #10/11 at 1/5 coverage) |
| 2 | InTheWild | 29.725 | 0 | 0.0e+00 | #12 | ✅ | ✅ | (batched) | LIVE |
| 3 | ASVspoof2019_LA | 9.439 | 0 | 0.0e+00 | #25 | ✅ | ✅ | (batched) | LIVE — silver tier @ 3/5 coverage |
| 4 | ASVspoof2021_LA | 32.056 | 0 | 0.0e+00 | #12 | ✅ | ✅ | (batched) | LIVE |
| 5 | ASVspoof2021_DF | 28.731 | 0 | 0.0e+00 | #12 | ✅ (3rd try) | ✅ | (batched) | LIVE |

**DF #12 CI note:** the 4 smaller datasets passed verify-pr green with identical lab260/AASIST3 hosting; only DF (611k trials, 3.4× next-largest) failed twice at ~200s on the documented un-retried label-stream ([[project_ci_429_burst_followup]]). A 3rd re-dispatch + the sweep backstop passed green. Result was verified-correct locally throughout (sha exact, reproduce --no-local Δ0.0).

## Final report

- **AASIST3 is LIVE on the Arena — 🥇 gold tier, rank #8 of 11, full 5/5 coverage.**
- All 5 dataset PRs merged; reproduction blocks filled (`match: scoring`).
- Backlink `result.yaml` for all 5 datasets + Arena badge block on the
  `lab260/AASIST3` model card. Benchmark dir README + implementation-notes written.
- Scores hosted in `lab260/AASIST3/.eval_results/` (per user; no new org repo).
- **Skipped:** none. **Systemic-bug flag:** none (EERs vary sensibly per domain).

| Dataset | EER % | Trials |
|---|---|---|
| ASVspoof2019_LA | 9.44 | 71,237 |
| ASVspoof2021_DF | 28.73 | 611,829 |
| InTheWild | 29.72 | 31,779 |
| CD-ADD | 30.73 | 20,786 |
| ASVspoof2021_LA | 32.06 | 181,566 |

**Backlinks + README/model-card badges:** batched at the END (one model-repo commit + cumulative tier/rank), since the cumulative badge reflects final coverage. genbacklink.py output verified byte-identical to the post-merge-badge comment's result.yaml.

(Self-check before each MR: sane EER, `n_skipped ≈ 0`, reproduce self-consistency.
Fail → SKIP this dataset, fill the reason, continue. Red verify-pr → SKIP, leave MR
open. Fill `reproduction: match: scoring` at merge.)

## Final report
- Live on Arena: <list + tier/rank from `/badge/aasist3/tier.json`>
- Skipped: <dataset + reason each>

## Notes / guideline discrepancies
- <record any official-doc inaccuracies; propose fix + ask before editing official docs>
