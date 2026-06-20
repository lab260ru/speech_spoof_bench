# Arena submission plan: XLSR-SLS

---

# Plan (reviewed at the 🚦 PLAN GATE — before any wrapper/compute)

- **Source repo:** <https://github.com/QiShanZhang/SLSforASVspoof-2021-DF>
- **Model name / slug:** XLSR-SLS / `xlsr-sls`  (slug lowercase)
  - The paper's model = wav2vec 2.0 **XLS-R 300M** front-end + **SLS** (Sensitive
    Layer Selection) classifier. Repo name (`SLSforASVspoof-2021-DF`) is a poor
    slug; using the descriptive `XLSR-SLS`, mirroring the precedent of naming
    W2V2-AASIST off its architecture rather than its repo (`SSL_Anti-spoofing`).
    Distinguishes it from the already-submitted W2V2-AASIST (same XLS-R backbone,
    different head).
- **Checkpoint:** the single pretrained SLS model from the repo's Google-Drive
  folder (`13vw_AX1jHdYndRu1edlgpdNJpCX8OnrH`), fine-tuned on ASVspoof2019 LA
  train; the paper's headline model (1.92% EER ASVspoof2021 DF, 7.46% InTheWild).
  - Download via `gdown` to `benchmarks/XLSR-SLS/<ckpt>.pth` in setup. **If Drive
    blocks the download (quota/permissions) → upfront BLOCKER, I will stop and
    ask** (the only checkpoint source the repo offers besides Baidu).
  - Build dependency `xlsr2_300m.pt` (base XLS-R 300M, fairseq, 3.6 GB) is already
    present at `benchmarks/W2V2-AASIST/xlsr2_300m.pt` — **reuse it** (used only to
    construct the architecture; all weights then overwritten by the SLS checkpoint).
- **Paper:** ACM MM 2024 — Zhang, Wen, Hu, "Audio Deepfake Detection with
  Self-Supervised XLS-R and SLS Classifier", DOI `10.1145/3664647.3681345`,
  OpenReview `acJMIXJg2u`. **No arXiv version exists** → set `paper.url` to the
  OpenReview/DOI link + bibtex; no `arxiv_id`. Peer-reviewed ⇒ **ranked-tier eligible**.
- **params_millions:** count after load (~318M; XLS-R 300M dominates, as W2V2-AASIST).
- **Date:** 2026-06-05

## Wrapper approach
- Base class: **`AntiSpoofingModel`** (batched) — clone of `benchmarks/W2V2-AASIST/`.
- **Bona-fide score** = `output[:, 1]`. Source `main.py`:
  `batch_score = (batch_out[:, 1])...` — index 1 = bona fide, **higher = bona fide,
  no sign flip**. (Output is `LogSoftmax`; log-prob of bonafide is monotonic in EER.)
- Input window: deterministic **first-64600-sample** crop (~4 s @ 16 kHz),
  tile-repeat if shorter — `pad_fixed`, identical to source
  `data_utils_SSL.py::pad` at eval (no random crop). **No resampling** (audio
  arrives at `expected_sample_rate = 16000`).
- Source files ported: `model.py` → `_net.py` (`SSLModel` + `Model`/SLS head). The
  XLS-R backbone load reuses W2V2-AASIST's **fairseq cross-version fix**
  (`merge_with_parent(Wav2Vec2Config(), cfg.model)` + `build_model`, task=None) —
  the stock `load_model_ensemble_and_task([xlsr2_300m.pt])` fails on fairseq 0.12.2.
  Load weights strictly in `load()`; `.to(device)` **before** `.eval()`.

## Datasets (dynamic discovery) + cycle order
- Discovered (`benchmarks/*/` dirs with `eval.yaml`): **ASVspoof2019_LA,
  ASVspoof2021_DF, ASVspoof2021_LA, CD-ADD, InTheWild** (5).
- **Cycle order (smallest/cheapest first, by n_trials):**
  1. **CD-ADD** (~20.8k)
  2. **InTheWild** (~31.8k) — paper anchor: official-keys EER 7.46%
  3. **ASVspoof2019_LA** (~71.2k) — in-domain
  4. **ASVspoof2021_LA** (~181.6k)
  5. **ASVspoof2021_DF** (~611.8k) — paper anchor: official-keys EER 1.92%
- Note (as for W2V2-AASIST): this benchmark uses **curated trial sets**, so absolute
  EER will differ from the paper's official-keys numbers; sanity is "single-digit on
  the in-domain/strong sets, not ~50% random nor ~(100−true) inverted", plus the
  reproduce self-consistency check.

## 🚦 PLAN GATE — present the above; await explicit OK. Build/compute nothing before this.

---

# Execution log (filled autonomously after approval)

## Setup (one-time)
- [x] gdown'd → `benchmarks/XLSR-SLS/MMpaper_model.pth` (1.3 GB; top-level "MMpaper_model.pth"
      = ACM MM paper model; the `XWSB/model_15_EER...` in the same folder is an unrelated
      SVDD-challenge blend, discarded). sha256 `0d315184aa8e6f017ea72c4d2458c11bae8f07fd743fe860a3aa932e36135fa6`.
- [x] Wrapper built; **5/5 unit tests pass** (incl. `batch_size > 1`; strict-load confirms
      architecture + `module.` DataParallel-prefix strip).
- [x] Datasets already registered locally (from W2V2-AASIST run); `local list` shows all 5; local-only.
- [x] Batch size tuned on RTX 4070 Ti SUPER (`CUDA_VISIBLE_DEVICES=1`): **`batch_size = 24`**
      (peak 123.4 utt/s; plateau from bs≈16). params = **340.79M**.
- [x] Model repo `SpeechAntiSpoofingBenchmarks/XLSR-SLS` created (public); checkpoint uploaded,
      model commit `7018a8128258fee17b1f8631f8f1fcbaf71d53c3`.

## Per-dataset cycle (ONE MR at a time; skip-and-continue on failure)

| # | Dataset | EER% | skipped | reproduce Δ | MR | verify-pr | merged | badge | outcome |
|---|---------|-----:|--------:|------------:|----|-----------|--------|-------|---------|
| 1 | CD-ADD | 9.806 | 0 | 0.0e+00 | #13 | ✅ | ✅ | ✅ | LIVE |
| 2 | InTheWild | 7.456 | 0 | 0.0e+00 | #11 | ✅ (re-dispatched) | ✅ | ✅ | LIVE |
| 3 | ASVspoof2019_LA | 0.231 | 0 | 0.0e+00 | #24 | ✅ | ✅ | ✅ | LIVE |
| 4 | ASVspoof2021_LA | 7.392 | 0 | 0.0e+00 | #11 | ✅ | ✅ | ✅ | LIVE |
| 5 | ASVspoof2021_DF | 3.931 | 0 | 0.0e+00 | #11 | ✅ | ✅ | ✅ | LIVE |

(Self-check before each MR passed for all 5: sane EER, `n_skipped = 0`, reproduce
self-consistency Δ = 0.0e+00 with full coverage. No skips. `reproduction: match:
scoring` filled at each merge.)

## Final report
- **Live on Arena: all 5/5 datasets.** `xlsr-sls` is **🥇 gold tier, rank #1 of 10**
  (`/badge/xlsr-sls/tier.json` → `"gold"`; `/badge/xlsr-sls/rank.json` → `"#1 of 10"`).
- **Skipped: none.** Every dataset computed, self-checked, verified ✅, merged, badged.
- Model repo `SpeechAntiSpoofingBenchmarks/XLSR-SLS` (public); model card uploaded
  (commit `2e8b862`); per-dataset scores + backlink `result.yaml` commit-pinned.
- Beats the same-front-end W2V2-AASIST on every out-of-domain set (DF 3.93 vs 8.32,
  CD-ADD 9.81 vs 38.57, InTheWild 7.46 vs 11.22); InTheWild reproduced the paper's
  headline 7.46 % exactly. No systemic-bug flag (all green, varied EERs as expected).

## Notes / guideline discrepancies
- **Schema vs published-no-arXiv paper.** `submission.schema.json` requires
  `system.paper.arxiv_id` (`minLength ≥ 1`) whenever a `paper` block is present, so
  a peer-reviewed paper with **no arXiv** (this case: ACM MM 2024) cannot be encoded
  cleanly. The Arena ranking (`ranking._systems_with_paper`) only needs `paper_url`
  *or* `arxiv_id` to grant ranked-tier eligibility — so this is a package-schema
  strictness, not an Arena requirement. **User decision (2026-06-05):** put the ACM
  **DOI** in `arxiv_id` (`10.1145/3664647.3681345`), `url` = the DOI link → ranked
  (gold), correct paper link; only a secondary Arena detail view mislabels it
  `arXiv:<doi>`. *Proposed fix (not applied — needs separate package change + Arena
  pin bump):* relax the schema so `paper` may be `url`-only with `arxiv_id` optional.
- **verify-pr transient drop.** InTheWild's `verify-pr` did not fire on first PR
  open; re-dispatched via a no-op commit to `refs/pr/N` (documented recovery) → ✅.
- **badge generated deterministically** via the package's own `badge.py` producers
  (`deterministic_badge.py`) rather than waiting on the queued `post-merge-badge`
  comment — byte-faithful to the workflow output.
