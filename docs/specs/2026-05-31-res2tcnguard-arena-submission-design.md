# Design: Add Res2TCNGuard to the Arena

**Date:** 2026-05-31
**Source model:** https://github.com/lab260ru/Res2TCNGuard
**Goal:** Wrap Res2TCNGuard as an anti-spoofing model, run it against all 5 core
Arena datasets (one at a time), and open a submission PR per dataset.

## Decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| HF model repo | `SpeechAntiSpoofingBenchmarks/Res2TCNGuard` (org-owned) |
| Paper representation | **Path A** — DOI `10.48084/etasr.8906` in the `arxiv_id` field (accepts cosmetic "arXiv:" mislabel in the Arena detail view) |
| Windowing | Deterministic first 64600 samples (tile-pad if shorter) |
| Inference | Batched `score_batch`; **`batch_size` chosen by a pre-inference sweep** (§0) |
| Worker tuning | Prototype a parallel-decode DataLoader (`num_workers` ≤ 16); if much faster, use a faithful custom runner (§0) |
| Wrapper + checkpoint location | `benchmarks/Res2TCNGuard/` (i.e. on the dataset drive) |
| Scope / order | All 5 core datasets, **ASVspoof2019_LA first** (in-domain sanity check) |
| CI mirror (`reproduce --no-local`) | **Skipped** — submit as-is, observe behavior |
| Datasets source | Local copies under `benchmarks/` (registered local datasets) |
| Between datasets | Open PR → hand link to user → user merges + checks Arena → confirm → next |
| GPU | `CUDA_VISIBLE_DEVICES=3` |

## Model facts (from `TCN.ipynb`)

- Architecture `TestModel`: SincConv front-end → Res2Net `Encoder` → dual `TemporalConvNet` → linear head. Pure PyTorch; the notebook's `dgl`/`torch_geometric` imports are training-only and **not** needed for inference.
- **Fixed-length input mandatory:** the head uses `nn.Linear(138,4)` / `nn.Linear(174,4)`, whose input dims depend on temporal length. Every clip must be padded/cropped to **64600 samples** (~4.04 s @ 16 kHz). Variable-length scoring is impossible.
- **Score direction:** eval uses `logits[:, 1]`, higher = bona fide — matches the package convention (`SimpleAntiSpoofingModel`/`AntiSpoofingModel`: higher = more bona fide, label 0 = bona fide).
- Checkpoint `best_1.495.pth` (818 KB), reported best EER **1.49%** on ASVspoof2019 LA eval. `params_millions` measured at load.
- Sample rate 16 kHz (`SincConv_fast(sample_rate=16000)`); the runner resamples inputs to `expected_sample_rate=16000`.

## Component 0 — Pre-inference performance sweep (once, before any dataset)

Goal: pick the fastest `batch_size` and decide whether a parallel-decode custom
runner is worth using. Run on GPU 3.

1. **Batch-size sweep (GPU throughput).** Load the model once; time `score_batch`
   over dummy `float32[64600]` tensors at batch sizes `[1,2,4,8,16,32,64]`
   (warm-up + median of N reps). Pick the batch size with the best utts/sec; set it
   as the `Res2TCNGuard.batch_size` class attribute.
2. **Worker sweep (decode throughput).** On a real slice of one local dataset
   (~1–2k rows), time a `torch.utils.data.DataLoader` that decodes + runs
   `_extract`/`_to_float32_mono_16k` (imported from the package — identical
   preprocessing) at `num_workers ∈ [0,2,4,8,16]` (**cap 16**). Record utts/sec.
3. **Decision.** Compare end-to-end throughput: official single-threaded runner vs
   a parallel-decode prototype at the best `num_workers`.
   - If the parallel path is **not materially faster**, use the official
     `speech-spoof-bench run` for all datasets (scores are canonical by definition).
   - If it **is** materially faster, build a small **faithful custom runner** that
     reuses the package's `_extract`/`_to_float32_mono_16k` and the model's
     `score_batch`, parallelizing only decode via the DataLoader. **Gate:** before
     using it for real, validate it produces the **same per-utterance scores** (by
     utt_id, within 1e-6) as the official runner on a subset; only then use it for
     full runs. EER is computed by the package from the resulting `scores.txt`
     either way.

Record the chosen `batch_size`, `num_workers`, and the runner decision in
`implementation-notes.md`.

## Component 1 — Wrapper `res2tcnguard.py`

Lives at `benchmarks/Res2TCNGuard/res2tcnguard.py` alongside the checkpoint
`benchmarks/Res2TCNGuard/best_1.495.pth`. Runs are launched with that dir on
`PYTHONPATH` (`--model-module res2tcnguard:Res2TCNGuard`).

Self-contained file holding the network classes copied verbatim from the notebook
(`SincConv_fast`, `Res2Block`, `SE_Block`, `Encoder`, `Chomp1d`, `TemporalBlock`,
`TemporalConvNet`, `TestModel`) plus the wrapper:

```python
class Res2TCNGuard(AntiSpoofingModel):
    name = "Res2TCNGuard"
    expected_sample_rate = 16000
    batch_size = <chosen by §0 sweep>

    def load(self):
        self.net = TestModel().eval()
        self.net.load_state_dict(torch.load(CKPT, map_location="cpu"))
        self.net.to(self.device)        # device from CUDA_VISIBLE_DEVICES=3

    @torch.no_grad()
    def score_batch(self, audios, srs):
        xs = [pad_fixed(a, 64600) for a in audios]   # first-64600, tile-pad if short
        x = torch.from_numpy(np.stack(xs)).to(self.device)
        _, logits = self.net(x)
        return logits[:, 1].cpu().tolist()           # higher = bona fide
```

`pad_fixed` = deterministic: `x[:64600]` if long enough, else tile-repeat to 64600
(the notebook's `pad`, not `pad_random`). No skips expected.

## Component 2 — HF model repo `SpeechAntiSpoofingBenchmarks/Res2TCNGuard`

- Upload `best_1.495.pth`.
- Per dataset, upload `scores.txt` to
  `.eval_results/SpeechAntiSpoofingBenchmarks/<DATASET>/scores.txt`, capture the
  commit SHA for the pinned `resolve/<sha>/...` URL.

## Component 3 — `meta.yaml`

`name="Res2TCNGuard"`, slug `Res2TCNGuard`, `code` = GitHub repo, `checkpoint` = HF
model-repo URL, `params_millions` = measured, and:

```yaml
paper:
  arxiv_id: "10.48084/etasr.8906"
  url: "https://etasr.com/index.php/ETASR/article/view/8906"
  bibtex: |
    @article{Borodin_Kudryavtsev_Mkrtchian_Gorodnichev_2024, ...}
```

## Per-dataset loop (5×, sequential)

Order: `ASVspoof2019_LA` → `ASVspoof2021_DF` → `ASVspoof2021_LA` → `CD-ADD` → `InTheWild`.
Manifest-pinned revisions (used for `dataset.revision` in each submission YAML):

| Dataset | revision |
|---|---|
| ASVspoof2019_LA | `9492c4a85ad91508b6da03c92c98c58aeaa02424` |
| ASVspoof2021_DF | `16d4f7d6c68694ac9b0bd43b83df322d1bc5102e` |
| ASVspoof2021_LA | `dc119733697c946fcd17fe7c1541d7f26b4bbe07` |
| CD-ADD | `c2de87d49b268b624e6af7440dc2890703098965` |
| InTheWild | `a957f2582802cdb5964e118818c2e46b3d61aa35` |

Steps per dataset (REVISED 2026-05-31 — reproduce-before-merge, see Addendum):
1. Run inference (uses the local registry copy; `PYTHONPATH=benchmarks/Res2TCNGuard`, `CUDA_VISIBLE_DEVICES=3`, `batch_size=4`): `speech-spoof-bench run --model-module res2tcnguard:Res2TCNGuard --datasets SpeechAntiSpoofingBenchmarks/<DATASET> --output-dir ./results` → `scores.txt` + `result.yaml`. (Canonical runner; Task 5 was dropped — GPU is the ~70 utt/s bottleneck.)
2. **Sanity gate (2019_LA only):** EER near 1.49% confirms wrapper + score direction. ~98% ⇒ direction flipped → fix before continuing. Out-of-domain datasets (the other four) are expected to show high EER — not a bug.
3. Upload `scores.txt` to the model repo (commit-pinned); capture SHA + `sha256sum`.
4. Author `submissions/<slug>.yaml` (slug = `res2tcnguard`, lowercase per schema) from `results_template.yaml`: `dataset.revision` = manifest-pinned revision, `scores` (eer/n_trials/n_skipped), `artifact.scores_url` (pinned), `scores_sha256`, `bench_version`.
5. **Reproduce as maintainer (BEFORE the PR):** `speech-spoof-bench reproduce <submission>.yaml --scoring --no-local` — re-downloads the score file, checks sha256, streams labels at the pinned revision, recomputes EER vs claimed (Δ must be ~0). On success, **fill the `reproduction:` block** in the YAML (`reproduced_by: SpeechAntiSpoofingBenchmarks`, `reproduced_at`, `reproduced_bench_version`, `match: scoring`).
6. `validate-submission` (offline schema check).
7. Open PR with the **filled** reproduction block: `hf upload SpeechAntiSpoofingBenchmarks/<DATASET> <slug>.yaml submissions/<slug>.yaml --repo-type dataset --create-pr`.
8. Hand PR link to user → user merges → system appears on the Arena automatically (webhook → re-ingest). Confirm, then proceed to next dataset.

## Risks tracked

- **InTheWild not in local registry** → `speech-spoof-bench local set` before its run (other 4 are registered).
- **Score direction** → caught by the 2019_LA sanity gate on dataset #1 (passed: EER 1.4956%).

## Addendum (2026-05-31): the `reproduction:` gate + how the Arena auto-updates

**What we learned the hard way.** Dataset #1's PR was merged with `reproduction: {}` empty.
The model then never showed on the Arena even after a forced Space restart. Root cause is
in `arena/ingest.py:_build_state()`:

```python
repro = sub.get("reproduction") or {}
if not repro.get("match"):
    warnings.append(Warning(..., reason="missing reproduction block — unverified, skipped"))
    continue                      # the submission is filtered OUT of the leaderboard
```

The Arena deliberately hides any submission whose `reproduction.match` is unset. It is a
**trust gate**, not a refresh-timing issue — no amount of refreshing surfaces an unverified
submission. Fix applied: a maintainer runs `reproduce --scoring` and fills the block; once
present, the submission appears on the next refresh.

**Does the Arena auto-update for a new model the way it does for a new dataset? Yes — it
already does, via the same webhook.** `arena/webhook.py` handles a dataset-repo `main`
update (PR merge) by scheduling `_refresh_and_commit(repo)`, which calls
`ingest.load_state(force_refresh=True)` and commits the new `cache.json`. So a merged
submission re-ingests automatically within a refresh cycle — identical to the
dataset-update path. The ONLY reason a new model doesn't "just appear" is the
`reproduction:` filter above. So the question reduces to: *who fills `reproduction`, and
when?*

Three ways to close that gap, in increasing automation:

1. **Status quo / current flow — maintainer fills before merge (CHOSEN).** Whoever uploads
   the scores acts as the reproducing maintainer: run `reproduce --scoring`, fill the block,
   open the PR already-verified. Merge → appears automatically. Keeps the human trust gate
   (a person ran the check and merged), zero new infrastructure. This is what we adopted for
   datasets 2–5.

2. **Auto-stamp in the post-merge workflow.** The `post-merge-badge` GitHub Action already
   runs on merge and the `verify-hf-pr` Action already ran `reproduce --scoring` on the PR.
   Extend post-merge to *write the `reproduction:` block back* to the merged YAML
   (`reproduced_by: <ci-bot>`, `reproduced_at: <date>`, `reproduced_bench_version`,
   `match: scoring`) and commit to `main`. The existing webhook then re-ingests and the model
   appears with no human edit — the closest to "datasets just appear." Trade-off: removes the
   human reproduction gate (acceptable because `reproduce --scoring` is deterministic and CI
   already gates the PR; the merge click remains the human checkpoint). Cost: a small change
   to `ci/post_merge_badge.py` + a commit-back token scope, plus care to avoid a
   webhook→commit→webhook loop (guard: skip if `reproduction.match` already set).

3. **Show unverified submissions, ranked separately.** Relax `ingest.py` to include
   submissions without `reproduction`, flagged "unverified" and excluded from ranked tiers.
   Weakest option — pollutes the board with unchecked numbers; not recommended.

**Recommendation:** keep option 1 for this effort; if the project later wants new models to
appear hands-off, implement option 2 (auto-stamp in post-merge) with the loop guard. No
Arena ranking-code change is needed for either.
