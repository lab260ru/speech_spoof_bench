# Phase 1 — ASVspoof2019_LA in v4 pointer form — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the 13 GB `ASVspoof2019_LA` HF dataset working copy to `drive3_8tb`, promote `PLAN.md` from v3 to v4, and migrate the dataset repo's submission format to the pointer-style v4 schema. Push the dataset changes to HF.

**Architecture:** No code is written — this is a documentation/conformance pass. Three artifacts change: (1) the project's `PLAN.md` and `ROADMAP.md` (in the pip-package git repo at `/home/kirill/speech-spoof-bench/speech-spoof-bench/`), (2) the dataset repo's `submissions/` directory (in the HF dataset git repo, post-move at `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/`), (3) the dataset working-copy's filesystem location.

**Tech Stack:** `rsync`, `git`, `huggingface-cli` (only for read-back sanity check), `python -c` with `datasets`. No new dependencies.

**Spec:** `docs/specs/2026-05-20-phase-1-asvspoof2019-la-v4-design.md` (commit `173becc`).

---

## Preconditions / Assumptions

- The user already has uncommitted Phase 0 box-tick edits in `docs/roadmap/ROADMAP.md` in the pip-package repo. **Do not blow these away.** They get amended into the ROADMAP edit in Task 3.
- The pip-package git repo (`/home/kirill/speech-spoof-bench/speech-spoof-bench/`) and the HF dataset git repo (`/home/kirill/ASVspoof2019_LA/`) are **separate, unrelated git repos**. Commits to one don't go to the other.
- HF remote credentials are already configured in `/home/kirill/ASVspoof2019_LA/.git/config` (verified: token-in-URL form, push works without prompt).
- The destination directory `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/` exists and is empty of the `ASVspoof2019_LA/` subfolder.

---

## File Structure

**Pip-package repo (`/home/kirill/speech-spoof-bench/speech-spoof-bench/`):**
- Modify: `docs/roadmap/PLAN.md` — promote title to v4, rewrite §1.6, patch §1.9 bullet, patch §5.1 table row.
- Modify: `docs/roadmap/ROADMAP.md` — update workspace-layout paragraph, tick Phase 1 checkboxes at the end.
- Create: `docs/plans/2026-05-20-phase-1-asvspoof2019-la-v4.md` (this file — already exists when the plan starts).

**HF dataset repo (post-move: `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/`):**
- Delete: `submissions/scores/` (currently contains only `.gitkeep`).
- Modify (overwrite): `submissions/README.md` — replace v3 prose with v4 HF-CLI submitter flow.
- Modify (overwrite): `submissions/results_template.yaml` — replace v3 schema with v4 pointer schema.

**Filesystem:**
- Delete: `/home/kirill/ASVspoof2019_LA/` (entire tree, after the move is verified).

---

## Task 1: Move the dataset working copy to drive3_8tb

**Files:**
- Source tree: `/home/kirill/ASVspoof2019_LA/` (13 GB)
- Destination tree: `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/`

- [ ] **Step 1: Confirm preconditions**

```bash
test -d /home/kirill/ASVspoof2019_LA/.git && echo "source ok"
test -d /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks && echo "dest parent ok"
test ! -e /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA && echo "dest empty ok"
git -C /home/kirill/ASVspoof2019_LA status --porcelain
```

Expected:
- All three `... ok` lines print.
- The fourth command (`git status --porcelain`) prints **no output** (working tree clean). If it does print anything, STOP — investigate before moving.

- [ ] **Step 2: rsync the working copy**

Run in foreground (no `&`, no `nohup`). Expect 10–30 minutes on a typical SATA/USB bridge.

```bash
rsync -aHAX --info=progress2 \
  /home/kirill/ASVspoof2019_LA/ \
  /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/
```

Trailing slashes matter — they copy the contents, not the source directory itself. `-aHAX` preserves perms, hardlinks, ACLs, xattrs.

Expected: rsync exits 0. Progress meter ends at ~13 GB / 100%.

- [ ] **Step 3: Verify integrity of the new copy**

Run each command. All four must succeed.

```bash
NEW=/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA

# 3a. git internals are intact
git -C "$NEW" fsck --full

# 3b. working tree matches the index
git -C "$NEW" status --porcelain

# 3c. HEAD matches the source HEAD
SRC_HEAD=$(git -C /home/kirill/ASVspoof2019_LA log -1 --format=%H)
NEW_HEAD=$(git -C "$NEW" log -1 --format=%H)
[ "$SRC_HEAD" = "$NEW_HEAD" ] && echo "heads match: $NEW_HEAD" || echo "MISMATCH"

# 3d. rsync dry-run with --checksum reports no differences
rsync -nai --checksum \
  /home/kirill/ASVspoof2019_LA/ \
  "$NEW/"
```

Expected:
- 3a: `git fsck` prints no errors (warnings about dangling commits are OK).
- 3b: empty output (clean tree).
- 3c: prints `heads match: <40-char-sha>`.
- 3d: empty output (no `>f.....` or similar lines).

If ANY of 3a–3d fails: STOP. Do not delete the source. Investigate manually.

- [ ] **Step 4: Delete the source**

Only after Step 3 passes all four checks.

```bash
rm -rf /home/kirill/ASVspoof2019_LA
test ! -e /home/kirill/ASVspoof2019_LA && echo "source removed"
```

Expected: `source removed` prints.

- [ ] **Step 5: Re-verify the new copy is still pushable**

```bash
git -C /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA remote -v
```

Expected: prints the `origin` URL pointing at `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA` with the embedded HF token. This means push will work without re-auth.

No commit in this task — nothing changed inside the git repo, only its filesystem location.

---

## Task 2: Promote PLAN.md to v4 (in pip-package repo)

**Files:**
- Modify: `/home/kirill/speech-spoof-bench/speech-spoof-bench/docs/roadmap/PLAN.md`

Four edits — apply each, then verify with grep.

- [ ] **Step 1: Title to v4**

Change line 1 from:
```
# SpeechAntiSpoofingBenchmarks — Full Infrastructure Spec v3
```
to:
```
# SpeechAntiSpoofingBenchmarks — Full Infrastructure Spec v4
```

- [ ] **Step 2: Replace the whole of §1.6 (lines ~192–242)**

Find the current heading `### §1.6 Submission format` and replace from that heading through the end of the v3 scores-file fenced block (the block that contains `LA_E_2834763 -1.234`). Replace with this exact content:

````markdown
### §1.6 Submission format

`submissions/<system-slug>.yaml`. Scores files do **not** live in the dataset repo. Each submission points to a pinned, commit-immutable URL in the submitter's own HF model repo, plus a sha256 of the file at that URL. The dataset repo only stores small YAML files; `validate-dataset` and `reproduce --scoring` verify the URL and sha at submission time and nightly thereafter.

```yaml
schema_version: 4

system:
  name: AASIST
  slug: aasist-clovaai-default
  description: Reference AASIST, default config, FP32.
  code: https://github.com/clovaai/aasist
  checkpoint: https://huggingface.co/<owner>/<aasist-repo>
  paper:
    arxiv_id: "2110.01200"
    url: https://arxiv.org/abs/2110.01200
    bibtex: |
      @inproceedings{jung2022aasist, ... }

dataset:
  id: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
  revision: 7f3a9b1c                       # commit sha of the dataset at scoring time
  split: test

scores:
  eer_percent: 0.83
  n_trials: 71237
  n_skipped: 0
  # Additional metric ids are added here as they're rolled out.

artifact:
  # Pinned, commit-immutable URL. Pattern:
  #   https://huggingface.co/<owner>/<repo>/resolve/<commit-sha>/.eval_results/<dataset-org>/<dataset-name>/scores.txt
  scores_url: https://huggingface.co/<owner>/<repo>/resolve/<sha>/.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt
  scores_sha256: 4f9b...
  bench_version: speech-spoof-bench==0.3.1

reproduction:
  reproduced_by: SpeechAntiSpoofingBenchmarks      # required, see §1.7
  reproduced_at: 2026-05-20
  reproduced_bench_version: speech-spoof-bench==0.3.1
  match: scoring                                    # "scoring" or "inference"

submitter:
  hf_username: kborodin
  contact: k.n.borodin@mtuci.ru

submitted_at: 2026-05-19
notes: "Reference implementation, default config, FP32."
```

**Rules:**
- `scores_url` MUST be pinned by commit sha (the `<sha>` in the `/resolve/<sha>/` segment of the URL). URLs that resolve via `main` or any branch ref are invalid — they're mutable and break sha verification.
- `scores_sha256` MUST be the sha256 of the file at `scores_url` at submission time. `speech-spoof-bench reproduce --scoring` (§2.5) fetches the URL and recomputes the sha to verify.
- If the model repo at `scores_url` is deleted or rewritten, the submission becomes irreproducible and is auto-flagged by `nightly-revalidate` (see §8 build order — CI/CD layer).
- The maintainer fills in the `reproduction:` block at merge time; submitters leave it empty.

**Scores file format** (uploaded by the submitter to their model repo, fetched from `scores_url`):
```
LA_E_2834763 -1.234
LA_E_1665632  2.871
```
One line per utterance. Higher = more bonafide.
````

- [ ] **Step 3: Patch §1.9 validation bullet**

In §1.9, find this bullet:

```
- Submission YAMLs in `submissions/` parse against the bundled JSON Schema, scores SHA matches the file, `reproduction:` block is present.
```

Replace with:

```
- Submission YAMLs in `submissions/` parse against the bundled JSON Schema, `scores_url` is reachable and pinned by commit sha, `scores_sha256` matches the file fetched from `scores_url`, `reproduction:` block is present.
```

- [ ] **Step 4: Patch §5.1 "What lives where" table row**

Find this row in the §5.1 table:

```
| `scores.txt` | Dataset repos, `submissions/scores/*.txt` | ~2 MB each (one number per utterance) |
```

Replace with:

```
| `scores.txt` | Submitter model repos at `.eval_results/<dataset-org>/<dataset-name>/scores.txt`; referenced by `scores_url` in each submission YAML | ~2 MB each (one number per utterance) |
```

- [ ] **Step 5: Verify no stray v3 references remain**

```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
grep -nE "scores_file|submissions/scores|schema_version: 1\b" docs/roadmap/PLAN.md
```

Expected: **empty output**. If anything is printed, it's a v3 residue that needs fixing before commit.

- [ ] **Step 6: Also verify the version bump landed**

```bash
head -1 docs/roadmap/PLAN.md
```

Expected: `# SpeechAntiSpoofingBenchmarks — Full Infrastructure Spec v4`.

No commit yet — bundles with Task 3 since they're one logical change.

---

## Task 3: Update ROADMAP.md and commit Phase 1 spec changes

**Files:**
- Modify: `/home/kirill/speech-spoof-bench/speech-spoof-bench/docs/roadmap/ROADMAP.md`

- [ ] **Step 1: Update the workspace-layout paragraph**

In `docs/roadmap/ROADMAP.md`, find this line (around line 13):

```
All repo working copies live directly under `/home/kirill/speech-spoof-bench/`. Datasets (parquet, audio) are produced/stored **outside** this folder (e.g. `/home/kirill/datasets/`) and pushed directly to HF — they're not git-tracked locally.
```

Replace `/home/kirill/datasets/` with `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/`:

```
All repo working copies live directly under `/home/kirill/speech-spoof-bench/`. Dataset working copies (parquet, audio, git history) live on the larger drive at `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/<dataset>/` and push directly to HF — they're not tracked in the project repo.
```

Also find this line lower down (around line 28):

```
Datasets-as-files live elsewhere (`/home/kirill/datasets/ASVspoof2019_LA_build/` for transient parquet, or just on HF). This folder holds **only code and config**.
```

Replace with:

```
Datasets-as-files live at `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/<dataset>/` (one git working copy per HF dataset) or just on HF. This folder holds **only code and config**.
```

- [ ] **Step 2: Tick the Phase 1 checkboxes**

In `## Phase 1 — Single dataset (...)`, change every `- [ ]` to `- [x]`. There are 7 boxes.

- [ ] **Step 3: Verify everything is staged correctly**

```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
git status -s
git diff --stat
```

Expected (order may vary, but should include):
- ` M docs/roadmap/PLAN.md`
- ` M docs/roadmap/ROADMAP.md`
- `?? docs/plans/2026-05-20-phase-1-asvspoof2019-la-v4.md` (this plan file, untracked)

If `docs/specs/2026-05-20-phase-1-asvspoof2019-la-v4-design.md` shows up, something is wrong — that file was already committed as `173becc`.

- [ ] **Step 4: Commit**

```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
git add docs/roadmap/PLAN.md docs/roadmap/ROADMAP.md docs/plans/2026-05-20-phase-1-asvspoof2019-la-v4.md
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
docs: promote spec to v4 (pointer-style submissions, drive3_8tb path)

- PLAN.md title v3 → v4
- PLAN.md §1.6 rewritten: scores files now live in submitter model
  repos and are referenced by pinned scores_url + scores_sha256
- PLAN.md §1.9 validation bullet updated for v4 fields
- PLAN.md §5.1 "what lives where" row updated
- ROADMAP.md workspace layout points to drive3_8tb
- ROADMAP.md Phase 1 boxes ticked
- Adds docs/plans/2026-05-20-phase-1-asvspoof2019-la-v4.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git log --oneline -3
```

Expected: new commit on top. Previous commit was `173becc` (spec). Now we have `<new-sha>` on top.

---

## Task 4: Delete `submissions/scores/` in the dataset repo

**Files:**
- Delete: `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/submissions/scores/`

- [ ] **Step 1: Inspect what's there**

```bash
DATASET=/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
ls -la "$DATASET/submissions/scores/"
```

Expected: only `.gitkeep` (0 bytes). If any other files exist, STOP — they're someone's real scores file and the spec says don't lose them.

- [ ] **Step 2: Remove the directory via git**

```bash
cd /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
git rm -r submissions/scores/
git status -s
```

Expected: `D  submissions/scores/.gitkeep` in the status output.

No commit yet — bundles with Task 7.

---

## Task 5: Rewrite `submissions/README.md` (dataset repo)

**Files:**
- Modify (overwrite): `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/submissions/README.md`

- [ ] **Step 1: Overwrite with v4 content**

Replace the entire file content with:

````markdown
# Benchmark Submissions

To submit a result, you'll upload two files (no git clone required):

1. **`scores.txt`** to **your own HF model repo** under `.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt`.
2. **`<your-slug>.yaml`** as a pull request to **this dataset repo** under `submissions/<your-slug>.yaml`.

The YAML in this repo carries a pinned URL pointing at your `scores.txt`, plus its sha256. Scores files do not live in this repo.

## Submitter workflow

### 1. Generate `scores.txt` locally

```bash
speech-spoof-bench run \
  --model-module <your_package>:<YourModelClass> \
  --datasets SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
```

Output: `results/ASVspoof2019_LA/scores.txt` (one line per utterance, `<utterance_id> <score>`, higher = more bonafide).

### 2. Upload `scores.txt` to your model repo

```bash
huggingface-cli upload <your-owner>/<your-model-repo> \
  results/ASVspoof2019_LA/scores.txt \
  .eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt \
  --repo-type=model \
  --commit-message="Add ASVspoof2019_LA scores"
```

**Note the commit sha** the CLI prints — you'll need it in the next step.

### 3. Fill in the submission YAML

Copy `results_template.yaml` to `<your-slug>.yaml` and fill in every field. The two most important fields:

- `artifact.scores_url`: the **pinned** URL to your uploaded scores file. Use the commit sha from step 2, not `main`:
  ```
  https://huggingface.co/<your-owner>/<your-model-repo>/resolve/<commit-sha-from-step-2>/.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt
  ```
  URLs with `/resolve/main/` are rejected because they're mutable.
- `artifact.scores_sha256`: `sha256sum results/ASVspoof2019_LA/scores.txt | awk '{print $1}'`.

Leave the `reproduction:` block empty — the maintainer fills it in at merge time.

### 4. Open the PR via HF CLI

```bash
huggingface-cli upload \
  SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA \
  <your-slug>.yaml submissions/<your-slug>.yaml \
  --repo-type=dataset \
  --create-pr \
  --commit-message="Add <your-slug> submission"
```

The CLI prints a PR URL. That's it.

### 5. Wait for maintainer reproduction

A maintainer runs `speech-spoof-bench reproduce --scoring <PR-branch>`, which:
- Fetches `scores_url`.
- Verifies the sha256 against `artifact.scores_sha256`.
- Recomputes EER from the file.
- Compares to your claimed `scores.eer_percent` (must match within 1e-6).

If it passes, the maintainer fills in `reproduction:` and merges. If it fails, you get a comment on the PR explaining why.

## Verification levels

| Level | What the maintainer checks | Cost |
|---|---|---|
| `scoring` (default) | sha + recomputed EER from your `scores.txt`. | Seconds. |
| `inference` (optional, follow-up) | Re-runs your checkpoint end-to-end and regenerates `scores.txt`. Must match within 0.05% EER. | Expensive. |

Submissions without a `reproduction:` block never appear in the arena.

## What about git clone + push?

You can do it that way too, but for a single 2 KB YAML it's massively heavier. The HF CLI path is the documented one.
````

- [ ] **Step 2: Verify the file looks right**

```bash
DATASET=/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
head -3 "$DATASET/submissions/README.md"
wc -l "$DATASET/submissions/README.md"
grep -c "scores_url" "$DATASET/submissions/README.md"
grep -c "submissions/scores" "$DATASET/submissions/README.md"
```

Expected:
- First line: `# Benchmark Submissions`.
- Line count: roughly 70–90.
- `scores_url` count: ≥ 2.
- `submissions/scores` count: 0 (no v3 references).

No commit yet — bundles with Task 7.

---

## Task 6: Rewrite `submissions/results_template.yaml` (dataset repo)

**Files:**
- Modify (overwrite): `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/submissions/results_template.yaml`

- [ ] **Step 1: Overwrite with v4 template**

Replace the entire file content with:

```yaml
schema_version: 4

system:
  name: ""
  slug: ""
  description: ""
  code: ""
  checkpoint: ""
  paper:
    arxiv_id: ""
    url: ""
    bibtex: |
      @article{...}

dataset:
  id: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
  revision: ""
  split: test

scores:
  eer_percent: 0.0
  n_trials: 71237
  n_skipped: 0

artifact:
  # Must be pinned by commit sha. Pattern:
  #   https://huggingface.co/<owner>/<repo>/resolve/<commit-sha>/.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt
  scores_url: ""
  scores_sha256: ""
  bench_version: ""

# Leave this block empty — the maintainer fills it in at merge.
reproduction:
  reproduced_by: ""
  reproduced_at: ""
  reproduced_bench_version: ""
  match: ""

submitter:
  hf_username: ""
  contact: ""

submitted_at: ""
notes: ""
```

- [ ] **Step 2: Verify it parses as YAML**

```bash
DATASET=/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
python3 -c "
import yaml
with open('$DATASET/submissions/results_template.yaml') as f:
    data = yaml.safe_load(f)
assert data['schema_version'] == 4
assert data['dataset']['id'] == 'SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA'
assert 'scores_url' in data['artifact']
assert 'scores_sha256' in data['artifact']
assert 'paper' in data['system']
assert 'arxiv_id' in data['system']['paper']
print('template parses, v4 fields present')
"
```

Expected: prints `template parses, v4 fields present`. Non-zero exit means malformed YAML — fix before continuing.

No commit yet — bundles with Task 7.

---

## Task 7: Commit and push dataset repo changes to HF

**Files:**
- Touched in this commit: `submissions/scores/.gitkeep` (deletion), `submissions/README.md`, `submissions/results_template.yaml`.

- [ ] **Step 1: Confirm clean staging**

```bash
cd /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
git status -s
```

Expected exactly three lines (order may vary):
```
D  submissions/scores/.gitkeep
 M submissions/README.md
 M submissions/results_template.yaml
```

If `README.md` (the top-level dataset README) or anything else appears, STOP — investigate.

- [ ] **Step 2: Stage and commit**

```bash
cd /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
git add submissions/README.md submissions/results_template.yaml
# scores/.gitkeep deletion was already staged by Task 4's `git rm`
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: migrate submission format to v4 pointer style

- Remove submissions/scores/ (scores now live in submitter model repos
  under .eval_results/<dataset-org>/<dataset-name>/scores.txt)
- Rewrite submissions/README.md with HF CLI submitter flow
- Rewrite submissions/results_template.yaml with v4 schema:
  schema_version=4, artifact.scores_url + scores_sha256, system.paper
  as a structured block

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git log --oneline -3
```

Expected: new commit on top of `06296dc`.

- [ ] **Step 3: Push to HF**

```bash
cd /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
git push origin master
```

Expected: `To https://...ASVspoof2019_LA  06296dc..<new-sha>  master -> master`.

If push fails with auth: the token in the remote URL has expired. Re-auth and retry — DO NOT --force.

- [ ] **Step 4: Sanity check that the dataset still loads from HF**

```bash
python3 -c "
from datasets import load_dataset
ds = load_dataset('SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA', split='test', streaming=True)
row = next(iter(ds))
assert set(row.keys()) == {'path', 'audio', 'label', 'notes'}
assert row['audio']['sampling_rate'] == 16000
print('load_dataset works:', row['path'], 'label=', row['label'])
"
```

Expected: prints `load_dataset works: LA_E_... label= 0` (or `1`). Streaming avoids re-downloading 13 GB.

If this fails: something about the push corrupted the dataset metadata — investigate before declaring done.

---

## Task 8: Phase 1 sign-off

**Files:** No code changes. Just verify the DoD from the spec's §7.

- [ ] **Step 1: Walk the spec's Definition of Done**

Run each check. All must pass.

```bash
# DoD-1: old source gone
test ! -e /home/kirill/ASVspoof2019_LA && echo "OK: source deleted"

# DoD-2: new working copy exists and is clean
NEW=/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
git -C "$NEW" status -s | wc -l  # expect: 0
git -C "$NEW" log --oneline -2

# DoD-3: HF web UI — manual visit
echo "Open in browser: https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/tree/main/submissions"
echo "Verify: no scores/ folder, README and template are v4."

# DoD-4: load_dataset works (already done in Task 7 step 4)
echo "load_dataset check: see Task 7 step 4 output"

# DoD-5: PLAN.md is v4
head -1 /home/kirill/speech-spoof-bench/speech-spoof-bench/docs/roadmap/PLAN.md
grep -c "schema_version: 4" /home/kirill/speech-spoof-bench/speech-spoof-bench/docs/roadmap/PLAN.md
grep -c "scores_url" /home/kirill/speech-spoof-bench/speech-spoof-bench/docs/roadmap/PLAN.md

# DoD-6: ROADMAP.md Phase 1 ticked
grep -A 10 "## Phase 1" /home/kirill/speech-spoof-bench/speech-spoof-bench/docs/roadmap/ROADMAP.md | grep -c "\- \[x\]"  # expect: 7
```

Expected:
- DoD-1 prints `OK: source deleted`.
- DoD-2: `0` lines of status output, log shows new commit on top of `06296dc`.
- DoD-3: manual visit confirms.
- DoD-4: previously printed `load_dataset works: ...`.
- DoD-5: `# SpeechAntiSpoofingBenchmarks — Full Infrastructure Spec v4`; `schema_version: 4` count ≥ 1; `scores_url` count ≥ 2.
- DoD-6: `7`.

- [ ] **Step 2: Push the pip-package repo changes (if remote is configured)**

```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
git remote -v
```

If a remote is configured, run `git push`. If not (the repo may be local-only at this stage of Phase 0), skip — those commits live locally until the GitHub remote is wired in a later phase.

- [ ] **Step 3: Announce completion**

Report:
- Source removed (Y/N).
- Push to HF succeeded with sha `<new-sha>`.
- `load_dataset` smoke test passed.
- PLAN.md/ROADMAP.md committed at `<sha>` in the pip-package repo.
- Phase 1 boxes ticked.

Phase 1 is done. Next phase is Phase 2 (pip-package skeleton), which has its own design + plan cycle.
