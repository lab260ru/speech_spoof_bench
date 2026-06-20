# Arena submission plan: Nes2Net (Nes2Net-X, wav2vec2 XLS-R)

- **Source repo:** <https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW> (speech repo;
  the user-supplied <https://github.com/Liu-Tianchi/Nes2Net> is the CtrSVDD *singing-voice*
  repo — wrong domain for this speech arena. User chose the speech repo at the gate.)
- **Model name / slug:** Nes2Net / `nes2net`
- **Checkpoint:** single **Nes2Net-X** (Google Drive id `1tjuSdbzgCnJSfy_eE_P52jRAonHY4YUT`),
  reported ASVspoof21 **LA 1.73% / DF 1.65%**. Local: `benchmarks/Nes2Net/nes2net_x_DF1.65.pth`.
  Alternative considered: 5-ckpt weight-average Nes2Net-X (DF 1.49% / LA 1.88%) — flag at Gate 1.
- **Paper:** Nes2Net, IEEE T-IFS 2025. arXiv 2504.05657, DOI 10.1109/TIFS.2025.3626963.
- **params_millions:** XLS-R 300M frontend + 0.511M Nes2Net-X backend (count exactly post-load).
- **Date:** 2026-06-04

## 0. Canon read
- [x] Read submit-model.md, new-model.md, testing-and-pitfalls.md
- [x] Studied reference model: `benchmarks/W2V2-AASIST/` (same XLS-R frontend + Tak lineage)

## 1. Wrapper (`benchmarks/Nes2Net/`)
- [x] `_net.py` — W2V2-AASIST SSLModel loader (fairseq merge_with_parent) + Nes2Net-X backend
      (Bottle2neck, SEModule, ASTP, Nested_Res2Net_TDNN) ported from source. Strict load: 0/0.
- [x] `nes2net.py` — `AntiSpoofingModel`: `load()`, `score_batch()`, `unload()`
  - [x] higher score = more bona fide (`logits[:,1]`; source label bonafide=1)
  - [x] weights loaded in `load()`, not `__init__`
  - [x] no resampling; deterministic `pad(64600)` window (source 4-sec eval protocol)
- [x] `meta.yaml`, `sweep.py`, `test_nes2net.py` (batch_size > 1) — 5/5 tests pass
- [x] symlink `xlsr2_300m.pt` from W2V2-AASIST (don't duplicate 3.6 GB)

## 2. Datasets (dynamic discovery, local-only)
- [x] Discovered (dirs with `eval.yaml`): ASVspoof2019_LA, ASVspoof2021_DF,
      ASVspoof2021_LA, CD-ADD, InTheWild
- [x] All already registered (`local list` confirms all 5)
- [ ] Never pass `--no-local`

## 3. Batch size (single RTX 4070 Ti SUPER, smi index 3)
- [x] `export CUDA_DEVICE_ORDER=PCI_BUS_ID; export CUDA_VISIBLE_DEVICES=3`
- [x] Ran `sweep.py`; peak bs=24 (124.4 utt/s; 16/24/32 plateau). `batch_size=24` set in wrapper.

## 4. Smoke validation (ASVspoof2019_LA — in-domain, most diagnostic)
- [x] EER = **0.128%**, n_skipped = **0**, full 71,237 coverage (in-domain, sane, not inverted)
- [x] `reproduce` self-consistency: sha match + EER recompute Δ = 0.0e+00 (PASS)

## 🚦 GATE 1 — present plan + wrapper + batch size + smoke result; await OK

## 5. Full run (all 5 datasets, local)
- [x] results/<DS>/ written for each; all reproduce PASS (Δ=0.0, sha ok, 0 skipped, full coverage)

  | dataset | EER% | n_trials | n_skipped | W2V2-AASIST (same bench) |
  |---------|-----:|---------:|----------:|-------------------------:|
  | ASVspoof2019_LA (in-domain) | 0.128 | 71,237 | 0 | 0.224 |
  | ASVspoof2021_LA | 6.141 | 181,566 | 0 | 8.113 |
  | ASVspoof2021_DF | 3.614 | 611,829 | 0 | 8.318 |
  | InTheWild | 8.476 | 31,779 | 0 | 11.222 |
  | CD-ADD | 20.549 | 20,786 | 0 | 38.569 |

## 🚦 GATE 2 — present all results; on OK, `submit` per dataset

## 6. Publish + PRs (manual upload path — submit would re-stream 900k+ files + local-registry 404)
- [x] Model repo `SpeechAntiSpoofingBenchmarks/Nes2Net`; checkpoint + 5 scores.txt in one
      commit, pinned sha `d41d1ebe4c0531c8e9717a5876c8da006cda45ff`
- [x] Built + schema-validated 5 submission yamls (manifest-pinned dataset revisions)
- [x] Opened 5 HF PRs: 2019_LA #23, 2021_LA #10, 2021_DF #10, InTheWild #10, CD-ADD #12
- [x] verify-pr CI: **all 5 ✅ PASS** (CD-ADD via webhook; other 4 via refs/pr/N re-dispatch
      after the burst hit the hf-ci 1+1 cap). NB: verdict bot posts AS `korallll`
      (HF_BOT_TOKEN = submitter token) — detect verdict by content, not author.
- [x] **All 5 merged** (reproduction block filled `match: scoring` on each PR branch pre-merge).

## 7. Badges (after each merge) — DONE
- [x] Per-dataset backlink `result.yaml` uploaded to model repo `.eval_results/<DS>/result.yaml` (×5)
- [x] Model-card README (all 5 EER badges + tier/rank) uploaded to `SpeechAntiSpoofingBenchmarks/Nes2Net`;
      local `benchmarks/Nes2Net/README.md` matches
- [x] post-merge-badge comment posted on 4/5 PRs; **ASVspoof2021_DF #10 badge comment dropped in the
      merge burst** (post-merge-badge has no sweep backstop) — cosmetic only; DF is fully merged,
      reproduction-filled, backlinked, and counted in the live rank.

## OUTCOME
**Nes2Net live on the Arena: 🥇 gold tier, rank #1 of 9.** Beats W2V2-AASIST on all 5 datasets.
Model repo `SpeechAntiSpoofingBenchmarks/Nes2Net`, scores pinned to commit `d41d1ebe`.

## Notes / guideline discrepancies
- Source `easy_inference_demo.py` '4s' mode uses 64000 samples, but the actual eval
  pipeline (`data_utils_SSL.Dataset_ASVspoof2021_eval`, default `--test_protocol 4sec`)
  uses **64600** — same as Tak/W2V2-AASIST. Using 64600 for all datasets (fixed window).
  ITW paper number used 'full' length; arena uses the fixed window for all (reproduce =
  self-consistency, not paper-number match).
