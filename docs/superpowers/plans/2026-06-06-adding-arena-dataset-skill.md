# `adding-arena-dataset` Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author a project-local `adding-arena-dataset` skill (`SKILL.md` + `plan-template.md`) that turns the proven raw-audio→Arena-dataset runbook (just exercised for ASVspoof5) into a reusable, one-gate procedure.

**Architecture:** Skill authoring follows skill-TDD (RED baseline → GREEN write → verify → REFACTOR), per `superpowers:writing-skills`. The skill is a *controllable runbook over the repo docs*, the dataset-side sibling of the existing `submitting-arena-model` skill, living at `speech-spoof-bench/.claude/skills/adding-arena-dataset/`.

**Tech Stack:** Markdown skill files; subagents for baseline + compliance testing; the `speech-spoof-bench` CLI / HF docs are the canon the skill points at.

---

## Reference facts (verified)

- Sibling skill (the pattern to mirror): `speech-spoof-bench/.claude/skills/submitting-arena-model/{SKILL.md,plan-template.md}`.
- Skill-authoring canon: `superpowers:writing-skills` — description = *when to use only* (no workflow summary); skill-TDD Iron Law (baseline first).
- Reference dataset dirs: `benchmarks/ASVspoof5/` (clean raw-embed path), `benchmarks/ASVspoof2021_LA/` (dirty re-encode path).
- Repo canon the skill cites: `speech-spoof-bench/docs/developing/new-dataset.md`, `docs/submitting/submit-dataset.md`, `docs/architecture/versioning.md`.
- Design spec: `speech-spoof-bench/docs/superpowers/specs/2026-06-06-adding-arena-dataset-skill-design.md`.
- Proven plan to generalize: `speech-spoof-bench/docs/superpowers/plans/2026-06-06-asvspoof5-dataset.md`.

## File structure

| File | Responsibility |
|------|----------------|
| `speech-spoof-bench/.claude/skills/adding-arena-dataset/SKILL.md` | The runbook: gate, pipeline, control rules, pitfalls |
| `speech-spoof-bench/.claude/skills/adding-arena-dataset/plan-template.md` | Two-half plan template (gate decisions + execution log) |

---

## Task 1: RED — baseline (how an agent does this WITHOUT the skill)

**Files:** none (produces notes used to shape the skill).

- [ ] **Step 1: Dispatch a baseline subagent (no skill)**

Dispatch a general-purpose subagent with this prompt and capture its full answer:

```
You are working in the Speech Anti-Spoofing Arena repo at /home/kirill/speech-spoof-bench.
A new raw dataset is at /data/example_eval (a folder of .flac + a whitespace TSV protocol
with an id column and a bonafide/spoof label column, ~600k clips on a spinning HDD).
Produce a concrete step-by-step plan to package it into an Arena-ready HF dataset and get it
onto the leaderboard. Be specific about: how you verify the audio is usable, how you choose
read/worker parameters, how you validate before and after publishing, and how it ends up
ranked. Do NOT execute anything — just write the plan.
```

- [ ] **Step 2: Record the baseline failures**

Write the agent's wrong/weak calls into a scratch list (used in Task 2). Expected gaps to look for (from lived experience):
1. Probes decodability on a **sample**, not the whole set (or skips it).
2. Throws max workers at a spinning disk / ignores **sorted-read** ordering.
3. Plans the **full online `validate-dataset`** (80 GB re-download) as the post-push check.
4. Omits the README `arxiv` front-matter key (D6 fail) / no DOI fallback.
5. Defaults to Extended or silently picks Core without flagging the **coverage** impact.
6. Forgets `labels.parquet`, or leaves the local `arena-manifest` clone dirty.
7. Doesn't put the working dir under `benchmarks/<Dataset>/`.

Expected: at least 3–4 of these appear. This confirms the skill teaches something non-obvious.

---

## Task 2: GREEN — write `SKILL.md`

**Files:**
- Create: `speech-spoof-bench/.claude/skills/adding-arena-dataset/SKILL.md`

- [ ] **Step 1: Write the skill file**

Create the file with exactly this content:

````markdown
---
name: adding-arena-dataset
description: Use when packaging or adding a new dataset into the Speech Anti-Spoofing Arena from a raw audio directory plus a label/protocol file, or when working under a new benchmarks/<Dataset>/. Triggers on a raw audio path plus "add/package/prepare this dataset for the arena".
---

# Adding a dataset to the Arena

## Overview

Drive a raw audio dir + its label/protocol into an Arena-ready HF dataset and a manifest PR.
**There is exactly ONE approval gate: the written plan.** Build, validate, push, and open the
PR autonomously after it. The authoritative shape + rules live in the repo docs; this skill is
the controllable runbook over them.

**Input:** a raw audio dir + a protocol/label source. Infer the rest: dataset name + **slug
(lowercase)**; the protocol column→field mapping and expected counts; license (confirm
redistribution); Core vs Extended (a deliberate, plan-time call).

## Read these first (do not skip)

- `speech-spoof-bench/docs/developing/new-dataset.md` — canonical repo shape, the D1–D7 checks
- `speech-spoof-bench/docs/submitting/submit-dataset.md` — publish + manifest-PR path
- `speech-spoof-bench/docs/architecture/versioning.md` — Core vs Extended, re-pinning
- A reference dataset dir: `benchmarks/ASVspoof5/` (clean **raw-byte embed** path) or
  `benchmarks/ASVspoof2021_LA/` (**re-encode** path for undecodable source FLAC)

## The single gate (the only routine stop)

| Gate | When | Present to user | Proceed on |
|------|------|-----------------|------------|
| **Plan** | Plan written, **before** building/probing/pushing | dataset name+slug, raw source paths, protocol column→field map + expected total/bonafide/spoof, license (+redistribution), **Core vs Extended** (flag the coverage impact), builder path (clean-embed vs re-encode, pending the probe), shard sizing | explicit OK |

After this gate, **do not ask for routine approval** — run the pipeline to the end, then report.
Stop only for an upfront blocker (see Control rules).

## Workflow

### Before the gate
1. **Read the canon** (above) + a reference dataset dir.
2. **Write the plan** by copying `plan-template.md` from this skill to
   `speech-spoof-bench/docs/plans/<YYYY-MM-DD>-<slug>-arena-dataset.md`; fill the upfront half.
3. **🚦 PLAN GATE.** Present it; proceed on explicit OK. Probe/build/push nothing first.

### After approval (autonomous)
Work under **`benchmarks/<Dataset>/`** (next to the other datasets — `benchmarks/` may be a
symlink to a big drive; that's fine, the parquet lands there).

  a. **Scaffold.** From `benchmarks/`: `speech-spoof-bench scaffold-dataset --name <N>
     --output-dir ./<N>`; copy `LICENSE.txt`/`.gitattributes`/`.gitignore` from a reference dir.
  b. **Probe decodability on EVERY clip** with soundfile (the decoder HF/validator/models use).
     Parallelise, but read in **sorted filename order** (spinning disk: ~7× vs random; worker
     count plateaus ~64). **0 failures → embed raw bytes (CLEAN, fast). Any failure → re-encode
     path** (librosa→clean 16 kHz FLAC, see `ASVspoof2021_LA`). A sample probe can lie — probe all.
  c. **Build.** Fork the reference `build_parquet.py`: 4 columns (`path`, `audio`@16k, `label`,
     `notes` JSON with a unique stable `utterance_id`); process **sorted by `utterance_id`**;
     resumable + atomic per shard; **delete stale `-of-NNNNN` shards** before writing; ~300–420
     MB/shard; assert expected counts; emit `data/labels.parquet` at the end.
  d. **Validate offline** until green: `speech-spoof-bench validate-dataset ./<N> --skip-submissions`.
     D6 needs an `arxiv` front-matter key — no arXiv? put the **DOI** there.
  e. **Push.** `create_repo(<org>/<N>, repo_type="dataset", exist_ok=True)` +
     `HfApi().upload_large_folder(...)` (resumable). Then a **fast online sanity check**:
     `load_dataset(..., streaming=True)` first rows decode + read `data/labels.parquet` counts.
     **Do NOT run the full online `validate-dataset`** — it downloads every shard (~80 GB / hours)
     and xet is content-addressed, so offline-green + this sanity check is sufficient. Capture the SHA.
  f. **Manifest PR.** Edit local `arena-manifest/manifest.yaml` (add to `core_set` or `extended`
     at the pinned **lowercase-hex** SHA); append a `dataset_added` `CHANGELOG.yaml` event;
     validate with `python -c "from speech_spoof_bench import manifest; manifest.load_manifest('manifest.yaml')"`;
     open the PR via `HfApi().create_commit(..., create_pr=True)`; then **revert the local clone**
     (the PR holds the change). No `schema_version`/`ranking_version` bump (data change).
  g. **Report** repo + SHA + PR URL; flag maintainer to-dos: review/merge (Core re-computes every
     submission's coverage), re-ingest to subscribe the webhook, fill the reproduction block when
     the first submission lands.

## Control rules

- **One gate.** The written plan is the only routine stop; run the pipeline to the end after OK.
- **Never touch the source dataset** — read-only; all output under `benchmarks/<Dataset>/`.
- **Probe the whole set**, never just a sample, before choosing the clean path.
- **Local-clone hygiene.** Revert `arena-manifest` working edits once the PR is open.
- **Stop only for upfront blockers** — no redistribution rights (license forbids it), a
  missing/garbled protocol or id↔file mismatch, or a broken env / HF auth. Ask, then continue.
- **Guideline fixes: propose, then ask** before editing official docs (`docs/developing/*`,
  `docs/submitting/*`). This skill file itself is the user's to edit on request.

## Common pitfalls (lived runs)

| Symptom | Cause / fix |
|---------|-------------|
| Sample probe clean, build ships undecodable audio | Decode-probe the **whole** set before raw-embed |
| Source read crawls | Spinning-disk random reads (~65 f/s) don't scale with workers; read **sorted by filename** (~440 f/s); ~64 workers plateaus |
| "Where did 80 GB go?" | `benchmarks/` may **symlink** to a big drive — parquet lands there (expected) |
| Publish step hangs for hours | Online `validate-dataset` downloads **every** shard; use streaming + `labels.parquet` sanity check instead |
| D6 fails: missing `arxiv` | Front-matter needs `arxiv`; no arXiv → put the **DOI** there |
| `_verify` count wrong after re-shard | Stale `-of-NNNNN` shards — delete mismatched-suffix shards first |
| Coverage shifts for every model | Adding to **Core** re-computes coverage — deliberate, plan-time; call it out in the PR |
| Manifest clone diverges / accidental push | Revert local `arena-manifest` edits after opening the PR |
| Re-shard breaks all submissions | `utterance_id` is the immutable join key — keep ids stable |

## Reference

`plan-template.md` (this dir) — instantiate at step 2; the upfront half is the gate, the
execution-log half is filled autonomously. Reference datasets: `benchmarks/ASVspoof5/`,
`benchmarks/ASVspoof2021_LA/`.
````

- [ ] **Step 2: Check description discipline + token budget**

Run:
```bash
cd /home/kirill/speech-spoof-bench/.claude/skills/adding-arena-dataset
python3 -c "import yaml,io; t=open('SKILL.md').read(); fm=t.split('---')[1]; d=yaml.safe_load(fm); print('name ok:', d['name']=='adding-arena-dataset'); print('desc starts Use when:', d['description'].startswith('Use when')); print('desc chars:', len(d['description']))"
wc -w SKILL.md
```
Expected: name ok True; desc starts True; desc < 500 chars; word count in the ~700–950 range (a detailed runbook, in line with the sibling skill). The description must NOT summarize the workflow (only triggers).

---

## Task 3: GREEN — write `plan-template.md`

**Files:**
- Create: `speech-spoof-bench/.claude/skills/adding-arena-dataset/plan-template.md`

- [ ] **Step 1: Write the template**

Create the file with exactly this content:

````markdown
# Arena dataset plan: <DATASET_NAME>

> Instantiate by copying this file to
> `speech-spoof-bench/docs/plans/<YYYY-MM-DD>-<slug>-arena-dataset.md` and filling every
> `<...>`. Delete this quote block.
>
> The **"Plan (reviewed at the gate)"** section is what the user approves — fill it *before*
> probing/building/pushing. Everything under **"Execution log"** is filled autonomously.

---

# Plan (reviewed at the 🚦 PLAN GATE — before any probe/build/push)

- **Dataset name / slug:** <Name> / <name-slug>  (slug MUST be lowercase)
- **Raw source (read-only):** audio dir <path>; protocol/label file <path>
- **Date:** <YYYY-MM-DD>

## Protocol → schema mapping
- id column → `utterance_id` (= audio filename stem): <which column>
- label column → bonafide/spoof: <which column + value meaning>
- extra `notes` fields to keep: <list, or "none">
- **Expected counts:** total <n> / bonafide <n> / spoof <n>

## License & redistribution
- SPDX / HF tag: <e.g. odc-by>; redistribution permitted: <yes — basis>

## Manifest placement
- **Core or Extended:** <choice>  (Core re-computes coverage for every existing submission)

## Build approach (confirm clean vs re-encode after the whole-set probe)
- Source SR / format: <e.g. 16 kHz mono FLAC>; clip duration range: <s>
- Path: <CLEAN raw-byte embed | re-encode (undecodable source)>  — finalize after the probe
- Shard sizing: ~<300–420> MB/shard → ~<n> shards; reference dir to fork: <benchmarks/...>

## 🚦 PLAN GATE — present the above; await explicit OK. Probe/build/push nothing before this.

---

# Execution log (filled autonomously after approval)

- [ ] Scaffolded at `benchmarks/<N>/`; LICENSE/.gitattributes/.gitignore copied
- [ ] **Whole-set decodability probe:** <fails>/<total> → path = <CLEAN | re-encode>
- [ ] Read params: <workers>, sorted order; build time <m>; shards <n>; size <GB>; counts asserted
- [ ] `validate-dataset ./<N> --skip-submissions` → D1–D7 green
- [ ] Pushed `SpeechAntiSpoofingBenchmarks/<N>` @ `<sha>`; online sanity (stream + labels.parquet) OK
- [ ] Manifest PR (`<core_set|extended>` @ `<sha>` + CHANGELOG `dataset_added`): <PR URL>; local clone reverted

## Maintainer to-dos (surface in final report)
- Review/merge the manifest PR (Core changes everyone's coverage)
- Re-ingest to subscribe the webhook; fill the reproduction block when the first submission lands

## Notes / guideline discrepancies
- <record any official-doc inaccuracies; propose fix + ask before editing official docs>
````

---

## Task 4: GREEN verify — compliance test (WITH the skill)

**Files:** none.

- [ ] **Step 1: Dispatch a fresh subagent given the skill**

Dispatch a general-purpose subagent. Paste the full contents of the new `SKILL.md` into the
prompt, then the SAME scenario from Task 1 Step 1, plus: "Follow this skill. Produce the plan
you would present at the gate, and list the key decisions for probing, read params, post-push
validation, and manifest placement."

- [ ] **Step 2: Verify the baseline failures are now fixed**

Confirm the agent's answer now: (1) probes **all** clips; (2) uses **sorted-order** reads, ~64
workers; (3) uses the **streaming sanity check**, not full online validate; (4) includes the
`arxiv`/DOI key; (5) treats **Core vs Extended** as a flagged gate decision; (6) emits
`labels.parquet` + reverts the clone; (7) works under `benchmarks/<Dataset>/`.
Expected: all 7 addressed. Record any that the agent still misses.

---

## Task 5: REFACTOR — close any remaining loopholes

**Files:**
- Modify: `speech-spoof-bench/.claude/skills/adding-arena-dataset/SKILL.md` (only if Task 4 found a miss)

- [ ] **Step 1: Patch the gaps**

For each item the Task-4 agent still missed, add an explicit line/row (Control rules or pitfalls
table) targeting that specific miss. If Task 4 was clean, make no change and note "no refactor
needed."

- [ ] **Step 2: Re-verify (only if you changed the skill)**

Re-run Task 4 Step 1 with the updated skill; confirm the previously-missed items are now handled.
Expected: all 7 addressed.

---

## Task 6: Deploy

**Files:** none new.

- [ ] **Step 1: Final quality checks**

Run:
```bash
cd /home/kirill/speech-spoof-bench/.claude/skills/adding-arena-dataset
ls -1   # SKILL.md + plan-template.md present
python3 -c "import yaml; d=yaml.safe_load(open('SKILL.md').read().split('---')[1]); print('frontmatter OK:', set(d)=={'name','description'})"
grep -c "🚦 PLAN GATE" SKILL.md plan-template.md   # gate present in both
```
Expected: both files present; frontmatter OK True; gate marker present.

- [ ] **Step 2: Commit**

```bash
cd /home/kirill/speech-spoof-bench/.claude/skills
git add adding-arena-dataset/SKILL.md adding-arena-dataset/plan-template.md 2>/dev/null || true
# .claude/skills may live in the speech-spoof-bench repo or be untracked; commit wherever it is tracked:
cd /home/kirill/speech-spoof-bench && git -C "$(git -C speech-spoof-bench rev-parse --show-toplevel 2>/dev/null || echo .)" status --short | grep adding-arena-dataset || true
```
If the skills dir is under a git repo, commit it there:
```bash
git add .claude/skills/adding-arena-dataset/
git commit -m "feat: add adding-arena-dataset skill (raw audio -> Arena dataset runbook)"
```
If `.claude/skills/` is not tracked by any repo, report that to the user and leave the files in
place (the sibling `submitting-arena-model` skill lives in the same dir — match whatever its
tracking is).
Expected: files committed, or a clear report that the skills dir is untracked.

---

## Self-review notes

- Spec coverage: location/files (Tasks 2–3), one gate + autonomous pipeline (SKILL.md body),
  whole-set probe / sorted reads / streaming sanity / arxiv-DOI / stale-shard / Core-coverage /
  symlink / revert-clone pitfalls (SKILL.md table + Task 1 baseline targets), `benchmarks/<Dataset>/`
  working dir (SKILL.md "After approval"), two-half template (Task 3) — all present.
- Skill-TDD honored: RED baseline (Task 1) precedes writing (Task 2); GREEN verify (Task 4);
  REFACTOR (Task 5).
- No placeholders: full file bodies inline; `<...>` tokens appear only inside the plan-*template*
  (they are the template's fill-ins, by design), not in the skill or the steps.
