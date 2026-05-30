# SpeechAntiSpoofingBenchmarks — Documentation

This is the documentation root for the **SpeechAntiSpoofingBenchmarks** project: an
open, reproducible leaderboard ("the Arena") for speech anti-spoofing / audio
deepfake-detection models, plus the tooling that feeds it.

If you are here to **submit** something, start with the step-by-step guides:

- [Submit a model](submitting/submit-model.md)
- [Submit a dataset](submitting/submit-dataset.md)

If you want to **understand how the whole thing works** or **develop/extend it**,
read on.

---

## The four repositories

The project is split across four sibling git repos (they are *not* one mono-repo —
each has its own remote and release cadence):

| Repo | Remote | What it is |
|------|--------|------------|
| `speech-spoof-bench/` | `github.com/lab260ru/speech_spoof_bench` | The pip package + CLI + GitHub Actions workflows. The "brain". |
| `arena/` | `huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena` | The Docker HF Space that renders the live leaderboard and hosts the webhook. |
| `arena-manifest/` | `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/arena-manifest` | One file (`manifest.yaml`): the source of truth for datasets, tiers, and ranking rules. |
| `dataset-builders/` | `github.com/lab260ru/bench_dataset_builder` | Per-dataset build scripts (currently a placeholder). |

Plus, on Hugging Face, an open-ended set of **dataset repos**
(`huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/<name>`) and contributor-owned
**model repos** that hold the actual score files.

> ⚠️ The pip package lives under the **`lab260ru`** GitHub user, **not** under the
> `SpeechAntiSpoofingBenchmarks` org. Everything else is on Hugging Face under the org.

---

## Documentation map

### Architecture & reference — *how it works today*

| Doc | Covers |
|-----|--------|
| [architecture/overview.md](architecture/overview.md) | The system at a glance: all components, the end-to-end data flow, who talks to whom. **Start here.** |
| [architecture/package.md](architecture/package.md) | Internals of the `speech_spoof_bench` package: the model interface, the benchmark/runner, the dataset loader, the metric registry, EER. |
| [architecture/submission-lifecycle.md](architecture/submission-lifecycle.md) | The three YAML schemas (meta → result → submission), the full submit→verify→merge→badge chain. |
| [architecture/arena.md](architecture/arena.md) | The Arena Space: ingest, ranking algorithm, tiers, leaderboard tables, charts, the cache, cold-start. |
| [architecture/cicd.md](architecture/cicd.md) | The webhook → GitHub Actions pipeline, the three workflows, every secret/env var, when each fires. |
| [architecture/badges.md](architecture/badges.md) | Static and dynamic badges: colour thresholds, the live tier/rank endpoints. |
| [architecture/versioning.md](architecture/versioning.md) | **The authoritative version map.** Every version number in the system, what it governs, what bumps it, and *when* you must update it. |

### Developing & contributing — *how to change it without breaking it*

| Doc | Covers |
|-----|--------|
| [developing/setup.md](developing/setup.md) | Local dev environment, editable installs, the local dataset registry, fully-offline workflow. |
| [developing/new-model.md](developing/new-model.md) | Developing a new model wrapper and proving it works *before* you submit. |
| [developing/new-dataset.md](developing/new-dataset.md) | Building a new dataset from scratch: scaffold → parquet → validate-green → publish → manifest. |
| [developing/new-metric.md](developing/new-metric.md) | Adding a new metric to the package (e.g. min-tDCF). |
| [developing/contributing-package.md](developing/contributing-package.md) | Working on the package itself: tests, schema bumps, releases, and the pin dance. |
| [developing/arena-dev.md](developing/arena-dev.md) | Running and changing the Arena Space locally, then deploying. |
| [developing/testing-and-pitfalls.md](developing/testing-and-pitfalls.md) | **"It worked on my machine."** The verification matrix and the full catalogue of things that silently break. |

### Project history & planning

- [roadmap/ROADMAP.md](roadmap/ROADMAP.md) — the phased plan (Phases 0–13). The project is at Phase 10.
- [roadmap/PLAN.md](roadmap/PLAN.md) — the original design plan.
- `specs/` and `plans/` — per-phase design specs and execution plans.

---

## The one-paragraph summary

A contributor wraps their model in a tiny Python class, runs `speech-spoof-bench run`
to produce a `scores.txt`, uploads it to their own HF model repo, and opens a PR on a
dataset repo containing a small **pointer YAML**. A Hugging Face webhook pings the
Arena Space, which dispatches a GitHub Action that **re-verifies the scores**
(`reproduce --scoring`: re-download, check the SHA-256, recompute the metric) and
comments the verdict on the PR. A maintainer merges; another Action posts a paste-ready
**badge** snippet. The Arena refreshes, reads the `arena-manifest` to learn the ranking
rules, and re-ranks every system into tiers. Nothing is trusted on faith — every number
on the board has been recomputed from an immutable, commit-pinned score file.
</content>
