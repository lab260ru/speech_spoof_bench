# `adding-arena-dataset` Skill — Design

**Date:** 2026-06-06
**Goal:** Wrap the proven "package a raw dataset into an Arena-ready HF dataset and add it
to the manifest" procedure (just exercised end-to-end for ASVspoof5) into a reusable,
project-local skill — the dataset-side sibling of `submitting-arena-model`.

## Decisions (locked with user)

| Decision | Choice |
|----------|--------|
| Generality | **Any raw-audio → Arena dataset** (not ASVspoof-specific). Builder specifics are per-dataset; the skill encodes the process + gotchas. |
| Gates | **One** routine stop: the written plan. After approval, run build→validate→push→PR autonomously, then report. (Mirrors `submitting-arena-model`.) |
| Location | Project-local: `speech-spoof-bench/.claude/skills/adding-arena-dataset/` |
| Files | `SKILL.md` + `plan-template.md` (same two-file shape as `submitting-arena-model`) |
| Dataset working dir | **`benchmarks/<Dataset>/`**, alongside the existing datasets — same convention as the model skill's `benchmarks/<Model>/`. The skill scaffolds, builds, and validates there; that dir is what gets pushed to HF. |

## Why a skill (not just docs)

The repo docs (`docs/developing/new-dataset.md`, `docs/submitting/submit-dataset.md`,
`docs/architecture/versioning.md`) explain the *canonical shape* and *rules*. They do **not**
encode the operational runbook or the lived-experience gotchas (decodability-across-the-whole-set,
HDD sorted-read trick, the 80 GB online-validate trap, the symlink, the Core-coverage gate).
The skill is the *controllable runbook over those docs* — it points at them as canon and adds
the process discipline + pitfalls, exactly as `submitting-arena-model` does for models.

## SKILL.md structure (mirror `submitting-arena-model`)

1. **Frontmatter** — `name: adding-arena-dataset`; `description` triggering on "add/package a
   new dataset into the Arena" from a raw audio dir + label/protocol source, or working under a
   new `benchmarks/<Dataset>/`.
2. **Overview** — one approval gate (the plan); autonomous after.
3. **Read these first** — `docs/developing/new-dataset.md`, `docs/submitting/submit-dataset.md`,
   `docs/architecture/versioning.md`, and a reference dataset dir (`benchmarks/ASVspoof2021_LA/`
   for the dirty/re-encode path; `benchmarks/ASVspoof5/` for the clean raw-embed path).
4. **The single gate** — a table: the written plan, presented before building/pushing, proceed
   on explicit OK.
5. **Workflow** — Before the gate (read canon, write plan, gate) / After approval (the autonomous
   pipeline below).
6. **Control rules** — one gate; what counts as an upfront blocker (no redistribution rights,
   missing/garbled protocol, broken env / HF auth); local manifest clone reverted after PR;
   guideline-fix etiquette (propose then ask for official docs).
7. **Common pitfalls** — the lived-experience table (below).
8. **Reference** — `plan-template.md`; reference dataset dirs.

## The autonomous pipeline (after the plan gate)

Generalized from the ASVspoof5 run:

1. **Scaffold** — from the `benchmarks/` dir, `speech-spoof-bench scaffold-dataset --name <N>
   --output-dir ./<N>` so the working dir is **`benchmarks/<N>/`** next to the other datasets;
   copy `LICENSE.txt` / `.gitattributes` / `.gitignore` from a reference dataset dir. (Note:
   `benchmarks/` may be a symlink to a big drive — see pitfalls; that's fine and intended.)
2. **Probe decodability on ALL clips** (not a sample) with soundfile — the exact decoder HF
   `datasets`/validator/models use. **0 failures → CLEAN raw-byte embed; any failure → re-encode
   path** (librosa→clean FLAC, à la `ASVspoof2021_LA`). This is the make-or-break correctness
   decision.
3. **Tune read params, then build** — resumable, atomic per-shard (`os.replace`), target
   ~300–420 MB/shard, process **sorted by `utterance_id`** (fast reads + stable shards). Build
   **auto-removes stale `-of-NNNNN` shards** before writing. Emit `data/labels.parquet` at the end.
   In-script asserts enforce expected row/label counts.
4. **Validate offline** — `validate-dataset ./<N> --skip-submissions` until D1–D7 green. Iterate
   on the local dir (fast, offline).
5. **Push** — `create_repo(..., exist_ok=True)` + `HfApi().upload_large_folder(...)` (resumable);
   then a **fast online sanity check** (`load_dataset(..., streaming=True)` first rows decode +
   read `data/labels.parquet` counts) — **NOT** the full online `validate-dataset` (it downloads
   every shard, ~80 GB / ~2 h, redundant since xet is content-addressed). Capture the commit SHA.
6. **Manifest PR** — edit local `arena-manifest/manifest.yaml` (add to **`core_set` by default**;
   `extended` only if explicitly requested) at the pinned SHA, append a `dataset_added`
   `CHANGELOG.yaml` event, validate with `manifest.load_manifest`, open the PR via
   `HfApi().create_commit(..., create_pr=True)`, then **revert the local clone** (PR holds the
   change). No `schema_version`/`ranking_version` bump (data change). Revision must be lowercase
   hex 7–40.
7. **Seed the random baseline** — give the new dataset its first leaderboard row + an end-to-end
   submission-path smoke test. Submit the package's random baseline
   (`speech_spoof_bench.examples.random_baseline:RandomBaseline`, model repo
   `SpeechAntiSpoofingBenchmarks/random-baseline-asas`, EER≈50%) against the new dataset **via the
   `submitting-arena-model` skill** (don't duplicate its compute→MR→verify→merge→reproduction
   runbook). Every existing Core dataset already carries a `submissions/random-baseline.yaml`; this
   matches the convention. Note: the submission's `verify-pr` only routes once the dataset is
   ingested, so the verify/merge half may wait on the maintainer's merge + re-ingest.
8. **Report** — repo + SHA + manifest PR URL + baseline submission PR URL; flag maintainer to-dos:
   review/merge the manifest PR (Core re-computes coverage), re-ingest to subscribe the webhook,
   merge the baseline submission + fill its reproduction block.

## Common pitfalls table (lived experience → baked into SKILL.md)

| Symptom / trap | Cause / fix |
|---|---|
| Sample probe clean but build ships undecodable audio | A sample can lie — decode-probe the **whole** set before choosing raw-embed |
| Build crawls reading source | Spinning-disk **random** reads (~65 f/s) don't scale with workers; read in **sorted filename order** (~440 f/s, 7×); ~64 workers is the plateau |
| "Where did 80 GB go?" | `benchmarks/` (or the dataset dir) may **symlink** to a big drive (e.g. drive3_8tb) — parquet lands there |
| Publish step hangs for hours | Online `validate-dataset` **downloads every shard** (~80 GB); use a streaming + `labels.parquet` sanity check instead |
| D6 fails: missing `arxiv` | Front-matter needs an `arxiv` key; no arXiv → put the **DOI** there (see `reference_paper_no_arxiv`) |
| `_verify` count wrong after re-shard | Stale `-of-NNNNN` shards from a prior run — build must delete mismatched-suffix shards first |
| Coverage shifts for every model | Datasets default to **Core** (re-computes coverage) — expected; call it out in the PR. Extended only if explicitly requested |
| New dataset has an empty board | Seed it: submit the random baseline (`…random_baseline:RandomBaseline`) via `submitting-arena-model`; every Core dataset carries `submissions/random-baseline.yaml` |
| Local manifest clone diverges / risk of direct push | Revert the local `arena-manifest` edits after the PR is opened |
| Re-shard breaks all submissions | `utterance_id` is the immutable join key — keep ids stable across re-shards |

## plan-template.md

Two halves (same split as the model skill's template):
- **Upfront decisions (the gate reviews this):** dataset Name (source casing, e.g. `ASVspoof5` — = dir + HF repo + manifest id, no lowercase slug); raw source paths
  (read-only); protocol column→field mapping + expected total/bonafide/spoof counts; license
  (SPDX + redistribution confirmation); manifest set (**Core by default**); builder path
  (clean-embed vs re-encode, pending the probe); shard sizing; reference dataset dir to copy from.
- **Execution log (filled autonomously):** probe result; chosen read params + build time; offline
  validate result; HF repo + SHA; online sanity result; manifest PR URL; **random-baseline
  submission PR URL**; maintainer to-dos.

## Non-goals / YAGNI

- Not a generic "any HF dataset" tool — scoped to this Arena's schema + manifest.
- No automated merge / re-ingest (maintainer actions, by design).
- No re-encode implementation detail beyond pointing at `ASVspoof2021_LA` as the reference.

## Success criteria

The skill faithfully encodes the proven runbook, points at the correct repo canon, enforces the
single plan gate, and surfaces the lived-experience gotchas — such that a fresh agent can take a
new raw dataset to a Core/Extended manifest PR without re-discovering them. Authoring follows the
`superpowers:writing-skills` guidance.
