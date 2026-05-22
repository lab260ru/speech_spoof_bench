# Phase 3 — First manual submission

**Date**: 2026-05-21
**Scope**: Roadmap Phase 3 (`docs/roadmap/ROADMAP.md` §Phase 3)
**Spec reference**: `docs/roadmap/PLAN.md` §1.6, §1.7

## Goal

Produce a single, working `submissions/<slug>.yaml` in the `ASVspoof2019_LA` dataset repo, **hand-authored** (no `submit` CLI yet). This proves the v4 pointer-style schema works end-to-end before automating it in Phase 7.

## Non-goals

- No `submit` CLI, no JSON Schema validator, no `reproduce --scoring` automation — Phase 7.
- No PR/CI flow — Phase 8. We push to `main` directly because we are the maintainer.
- No badge generation — Phase 9.
- No second model or dataset — Phase 11/12.

## Architecture

Two HF repos are involved. The scores file lives in the **model repo** under a per-dataset folder; the dataset repo holds only a small YAML pointer.

```
SpeechAntiSpoofingBenchmarks/random-baseline-asas/    ← model repo (already created)
└── .eval_results/
    └── SpeechAntiSpoofingBenchmarks/
        └── ASVspoof2019_LA/
            └── scores.txt                            ← uploaded in step 3

SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/         ← dataset repo
└── submissions/
    └── random-baseline.yaml                          ← authored in step 5
```

The dataset YAML carries:
- `scores_url`: pinned by commit sha of the model repo (`/resolve/<sha>/...`)
- `scores_sha256`: sha256 of the file at that URL
- `reproduction`: filled in by the maintainer (us) at merge time

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Model repo owner | `SpeechAntiSpoofingBenchmarks` (org) | Keeps namespace inside the org; cleaner cross-references. |
| Model repo name | `random-baseline-asas` | Matches the roadmap convention; "asas" = anti-spoofing arena submission. |
| Submission slug | `random-baseline` | Short, matches the spec's example style. |
| Paper field | ASVspoof 2019 paper (arxiv 1911.01601) | Random baseline has no paper; cite the dataset paper per Phase 3 convention. Document in `notes`. |
| Determinism | Re-run and diff before uploading | `RandomBaseline` already uses `seed=0`; verify scores.txt is byte-identical to the existing one. |
| Repo creation | User creates HF repo via web UI | Already done by user before execution. |

## Execution steps

### Step 1 — Verify scores.txt determinism

Re-run `speech-spoof-bench run` with the same arguments as Phase 2, output to a fresh directory, then diff against `results/ASVspoof2019_LA/scores.txt`. Must match byte-for-byte. If not, stop and fix runner row-order before proceeding.

### Step 2 — User creates model repo

Already complete: `SpeechAntiSpoofingBenchmarks/random-baseline-asas` exists on HF.

### Step 3 — Upload scores.txt and capture commit sha

Use `huggingface_hub.HfApi.upload_file` (or `huggingface-cli upload`) to push the verified scores.txt to:

```
.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt
```

Capture the resulting commit OID from the upload return value. This is the `<sha>` that will be substituted into `scores_url`.

Also write a minimal `README.md` to the model repo so it isn't completely empty, and add a `.gitattributes` if the file size would otherwise trigger LFS (1.5 MB is well under HF's 10 MB threshold — likely no LFS needed).

### Step 4 — Compute sha256

Already known from Phase 2's `result.yaml`:

```
scores_sha256: 71ac000c0712a4551873dba87183e746cb9730cd5ab17aaa87892009bde55587
```

Recompute locally as a sanity check (`sha256sum scores.txt`).

### Step 5 — Hand-author submissions/random-baseline.yaml

Fill `results_template.yaml` in the LA working copy. Final content:

```yaml
schema_version: 4

system:
  name: random-baseline
  slug: random-baseline
  description: >
    Reference random baseline. Returns N(0, 1) for every utterance using a
    fixed seed (seed=0). EER ≈ 50% by construction. Used as the seeded
    smoke-test baseline for the arena (roadmap Phase 3).
  code: https://github.com/SpeechAntiSpoofingBenchmarks/speech-spoof-bench
  checkpoint: https://huggingface.co/SpeechAntiSpoofingBenchmarks/random-baseline-asas
  paper:
    arxiv_id: "1911.01601"
    url: https://arxiv.org/abs/1911.01601
    bibtex: |
      @article{wang2020asvspoof,
        title={ASVspoof 2019: A large-scale public database of synthesized,
               converted and replayed speech},
        author={Wang, Xin and Yamagishi, Junichi and Todisco, Massimiliano and
                Delgado, H{\'e}ctor and Nautsch, Andreas and Evans, Nicholas
                and Sahidullah, Md and Vestman, Ville and Kinnunen, Tomi and
                Lee, Kong Aik and others},
        journal={Computer Speech \& Language},
        volume={64},
        pages={101114},
        year={2020},
        publisher={Elsevier}
      }

dataset:
  id: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
  revision: 151aa4c6        # LA repo HEAD at submission time
  split: test

scores:
  eer_percent: 49.870836165873556
  n_trials: 71237
  n_skipped: 0

artifact:
  scores_url: https://huggingface.co/SpeechAntiSpoofingBenchmarks/random-baseline-asas/resolve/<UPLOAD_SHA>/.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt
  scores_sha256: 71ac000c0712a4551873dba87183e746cb9730cd5ab17aaa87892009bde55587
  bench_version: speech-spoof-bench==0.1.0

reproduction:
  reproduced_by: SpeechAntiSpoofingBenchmarks
  reproduced_at: 2026-05-21
  reproduced_bench_version: speech-spoof-bench==0.1.0
  match: scoring

submitter:
  hf_username: SpeechAntiSpoofingBenchmarks
  contact: k.n.borodin@mtuci.ru

submitted_at: 2026-05-21
notes: >
  Random baseline has no associated paper. The paper field cites the
  ASVspoof 2019 dataset paper as a placeholder, per the Phase 3
  convention documented in docs/specs/2026-05-21-phase-3-first-submission-design.md.
```

`<UPLOAD_SHA>` is substituted with the commit OID captured in step 3.

### Step 6 — Push to LA dataset repo main

In `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/`:

```bash
git add submissions/random-baseline.yaml
git commit -m "feat: add random baseline submission (Phase 3)"
git push origin main
```

## Verification

After execution:
- `huggingface.co/SpeechAntiSpoofingBenchmarks/random-baseline-asas/blob/main/.eval_results/.../scores.txt` is reachable.
- The pinned `scores_url` resolves to the same file (HTTP 200, sha256 matches).
- `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/blob/main/submissions/random-baseline.yaml` is reachable and valid YAML.
- Manual EER recomputation from the fetched scores.txt against the dataset labels yields `49.870836165873556` (within 1e-6).

The Phase 6 end-to-end smoke test (later) will consume exactly this submission.

## Risks

| Risk | Mitigation |
|---|---|
| Re-run scores differ from Phase 2's | Diff first. If different, fix runner before uploading. |
| LFS auto-conversion on upload | 1.5 MB is under HF's LFS threshold; verify upload result. If LFS gets used, the pinned URL pattern still works (HF serves both transparently via `/resolve/`). |
| Wrong commit sha captured | `HfApi.upload_file` returns a `CommitInfo` with `oid`; use that, don't guess. |
| LA repo revision drifts during step 5 | Capture `git rev-parse HEAD` of the LA working copy *before* authoring; if it moved, re-pin. |
