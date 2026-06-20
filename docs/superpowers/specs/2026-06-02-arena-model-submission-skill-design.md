# Design: `submitting-arena-model` skill

Date: 2026-06-02
Status: approved (pending spec review)

## Problem

Adding a new anti-spoofing model to the Arena is a long, multi-step pipeline
(wrap → tune batch size → run 5 datasets → submit PRs → badges). The steps are
documented across several files (`docs/submitting/submit-model.md`,
`docs/developing/new-model.md`, `docs/developing/testing-and-pitfalls.md`) and
demonstrated by reference models (`benchmarks/Res2TCNGuard/`,
`benchmarks/ResCapsGuard/`). Driving it from raw instructions each time is
error-prone and produces inconsistent control behaviour (too many questions, or
too few).

Goal: a **project skill** that compresses this into a controllable runbook —
parameterized by a model source repo URL, autonomous except at two explicit
gates, and self-consistent with the existing docs and reference models.

## Decisions (from brainstorming)

1. **Two control gates total** (not per-MR). Everything else autonomous; always
   stop on a hard error or genuine ambiguity.
2. **Guideline fixes: propose-then-ask.** When Claude finds the official docs
   wrong, it reports the discrepancy + proposed fix and waits for approval
   before editing the official guideline. The discrepancy is meanwhile recorded
   in the per-model `implementation-notes.md`.
3. **Hybrid plan shape.** The skill embeds a generic, model-agnostic plan
   template AND has Claude instantiate a per-model plan file under
   `docs/plans/<date>-<model>-arena-submission.md`, which it then executes.
4. **Local-only evaluation, dynamic dataset set.** Datasets are read from
   `benchmarks/` via the local-dataset registry; never downloaded. The set of
   datasets is discovered at runtime (every `benchmarks/` subdir with an
   `eval.yaml`), not hardcoded — it may grow as datasets are added.

## Skill location & files

```
/home/kirill/speech-spoof-bench/.claude/skills/submitting-arena-model/
  SKILL.md            # runbook: triggers, the two gates, control rules, gotchas
  plan-template.md    # the generic per-model plan (copied + adapted each run)
```

Placed at the **workspace-root `.claude/skills/`** so it auto-loads anywhere in
this tree. (That root is not itself a git repo; if version control is wanted,
relocate into `speech-spoof-bench/speech-spoof-bench/` and symlink.)

## Inputs

- **Required:** model source repo URL (e.g. `https://github.com/lab260ru/ResCapsGuard`).
- **Inferred; ask only when blocked:** model name/slug (from repo name),
  checkpoint location, paper / arxiv / bibtex, `params_millions`.

## The flow Claude follows

1. **Read the canon.** `docs/submitting/submit-model.md`,
   `docs/developing/new-model.md`, `docs/developing/testing-and-pitfalls.md`,
   and a reference model dir (`benchmarks/Res2TCNGuard/` or
   `benchmarks/ResCapsGuard/`).
2. **Instantiate the plan.** Copy `plan-template.md` →
   `docs/plans/<date>-<model>-arena-submission.md`; adapt to this model.
3. **Build the wrapper.** `AntiSpoofingModel` subclass in `benchmarks/<Model>/`,
   honouring the three gotchas:
   - higher score = more bona fide (label 0 = bonafide, 1 = spoof);
   - load weights in `load()`, not `__init__`;
   - never resample (audio arrives at `expected_sample_rate`).
4. **Discover datasets dynamically + register locally (no download).** The set
   of evaluation datasets is whatever is present under `benchmarks/`, not a
   hardcoded list. A dataset dir is any subdir of `benchmarks/` containing
   `eval.yaml` (model dirs contain `meta.yaml` instead and are skipped). For
   each discovered dataset `<DS>`:
   `speech-spoof-bench local set SpeechAntiSpoofingBenchmarks/<DS> benchmarks/<DS>`.
   Verify with `speech-spoof-bench local list`. (Count may be more or fewer than
   the 5 arena-manifest core datasets; run on all that are present locally.)
5. **Tune batch size.** Run `sweep.py` on a **single RTX 4070 Ti Super**; pick
   the throughput peak. Pin the GPU: `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
   `CUDA_VISIBLE_DEVICES=1` (torch's default fastest-first order ≠ nvidia-smi;
   the 4070 Ti Supers are PCI indices 1 and 3).
6. **Smoke-validate** one dataset: sane EER, `n_skipped ≈ 0`, and
   `reproduce --scoring` matches. Run via local registry (no `--no-local`).
7. **🚦 GATE 1.** Present generated plan + wrapper + chosen batch size + smoke
   result. Wait for approval before the long runs.
8. **Run all discovered datasets** locally (registry; no `--no-local`). Write
   `results/<DS>/` and `submissions/<DS>/` for each.
9. **Publish artifacts to the org.** Push checkpoint + `scores.txt` to
   `SpeechAntiSpoofingBenchmarks/<Model>` on HF; local copy stays in
   `benchmarks/<Model>/`.
10. **🚦 GATE 2.** Present all dataset results; on approval, open one MR per
    dataset as a batch.
11. **Badges.** After each merge, add badge snippets to the model `README.md`.

## Control rules (baked into the skill)

- Two gates only (steps 7 & 10). Otherwise autonomous.
- Always stop on a hard error or genuine ambiguity; ask, then continue.
- Guideline inaccuracies: propose + ask before editing official docs; log in
  `implementation-notes.md` regardless.

## Known pitfalls to encode (from testing-and-pitfalls.md)

- Score direction flipped → EER ≈ (100 − true). Smoke check catches it.
- Double resampling → numbers drift, `reproduce` disagrees.
- Heavy work in `__init__` breaks `--model-module` discovery.
- `> 5%` skip rate aborts (TooManySkips).
- Batch-only bugs hidden by per-item fallback — test real `batch_size > 1`.
- `local set` canonical-id mismatch can 404 at submit; `local list`/`show` to
  verify before submitting.
- `local-datasets.yaml` is gitignored at the pip-repo root.

## Departure from raw wording

The original instruction said "ask to approve **each** MR." The Two-gate
decision approves all per-dataset MRs **as a batch** at Gate 2. Badges are still
added per-model after merges. Intentional.

## Out of scope

- Implementing `reproduce --inference` (not implemented upstream).
- Re-pinning datasets in the manifest.
- Changing the package schema or CI workflows.
