# Phase 6 Smoke Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the five Phase 6 smoke-test checks and confirm the LA → submission YAML → manifest → Arena chain works end-to-end.

**Architecture:** Three autonomous checks (sha parity, EER parity, scores regeneration) run as ad-hoc Python via the existing `speech-spoof-bench` CLI and `huggingface_hub`. Two checks require live HF mutations + browser verification by the user: one round-trip edit (bump EER, confirm visible, revert) and one fault-injection (push malformed YAML, confirm warning, delete).

**Tech Stack:** Python, `speech-spoof-bench` CLI, `huggingface_hub`, `numpy`, Gradio Space (browser).

**Spec reference:** `docs/specs/2026-05-22-phase6-smoke-test.md`

**Working directories:**
- Package: `/home/kirill/speech-spoof-bench/speech-spoof-bench`
- LA dataset working copy: `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA`
- Arena: `/home/kirill/speech-spoof-bench/arena`

**Known reference values** (from `submissions/random-baseline.yaml` and local `results/`):
- `scores_sha256`: `71ac000c0712a4551873dba87183e746cb9730cd5ab17aaa87892009bde55587`
- `eer_percent`: `49.870836165873556`
- `n_trials`: `71237`
- Model repo: `SpeechAntiSpoofingBenchmarks/random-baseline-asas`
- Model repo path: `.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt`
- Model repo commit: `f63c30bade6e2d059b2e805dea7a807f2f57e99a`

This plan produces no new source files — it is a verification runbook. No commits are made except where noted (HF dataset repo edits, with explicit reverts).

---

### Task 1: Check 1 — scores.txt sha parity (three-way)

**Goal:** confirm `sha256(local scores.txt) == sha256(model-repo scores.txt) == YAML.scores_sha256`.

- [ ] **Step 1: Compute sha of local file**

Run:
```bash
sha256sum /home/kirill/speech-spoof-bench/speech-spoof-bench/results/ASVspoof2019_LA/scores.txt
```
Expected: `71ac000c0712a4551873dba87183e746cb9730cd5ab17aaa87892009bde55587`

- [ ] **Step 2: Download model-repo scores.txt at the pinned commit and sha it**

Run:
```bash
python -c "
from huggingface_hub import hf_hub_download
import hashlib, pathlib
p = hf_hub_download(
    repo_id='SpeechAntiSpoofingBenchmarks/random-baseline-asas',
    filename='.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt',
    revision='f63c30bade6e2d059b2e805dea7a807f2f57e99a',
)
print(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest())
"
```
Expected: `71ac000c0712a4551873dba87183e746cb9730cd5ab17aaa87892009bde55587`

- [ ] **Step 3: Read sha from submission YAML and assert all three match**

Run:
```bash
python -c "
import yaml, pathlib
y = yaml.safe_load(pathlib.Path('/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/submissions/random-baseline.yaml').read_text())
print('YAML scores_sha256:', y['artifact']['scores_sha256'])
"
```
Expected: prints `71ac000c0712a4551873dba87183e746cb9730cd5ab17aaa87892009bde55587`.

PASS if all three sha values are identical. If any differ, STOP and investigate before continuing.

---

### Task 2: Check 1b — Re-run baseline and confirm bit-identical regeneration

**Goal:** seeded baseline produces same scores.txt sha on a fresh run.

- [ ] **Step 1: Re-run the baseline to a scratch output dir**

Run:
```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
speech-spoof-bench run \
    --model-module speech_spoof_bench.examples.random_baseline:RandomBaseline \
    --datasets SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA \
    --output-dir /tmp/phase6-rerun
```
Expected: completes without errors; writes `/tmp/phase6-rerun/ASVspoof2019_LA/scores.txt` + `result.yaml`.

- [ ] **Step 2: Compute sha and compare**

Run:
```bash
sha256sum /tmp/phase6-rerun/ASVspoof2019_LA/scores.txt
```
Expected: `71ac000c0712a4551873dba87183e746cb9730cd5ab17aaa87892009bde55587`

PASS if equal. If different, the seed is leaking or the iteration order changed — STOP.

Note: if `--output-dir` is not a supported flag on the current CLI, fall back to running from `/tmp/phase6-rerun` as cwd so the default `results/` dir lands there. Confirm CLI flags with `speech-spoof-bench run --help` before running.

---

### Task 3: Check 2 — EER parity

**Goal:** EER recomputed from local scores.txt matches `scores.eer_percent` in submission YAML exactly.

- [ ] **Step 1: Recompute EER from local scores.txt**

Run:
```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
python -c "
from datasets import load_dataset
from speech_spoof_bench.metrics.eer import compute_eer

ds = load_dataset('SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA', split='test')
labels = {row['path']: row['label'] for row in ds}

scores = {}
with open('results/ASVspoof2019_LA/scores.txt') as f:
    for line in f:
        utt, s = line.strip().split()
        scores[utt] = float(s)

r = compute_eer(scores, labels)
print(f'{r.value!r}')
"
```
Expected: `49.870836165873556` (MetricResult.value).

- [ ] **Step 2: Compare against YAML**

The expected value in `submissions/random-baseline.yaml` is `49.870836165873556`. Assert equality of full floats (no rounding tolerance — both derive from the same scores).

PASS if equal. If unequal, the YAML was hand-authored incorrectly OR EER implementation has drifted — STOP.

Note: confirm the score-file format (`utt score` whitespace-separated) by reading the first 2 lines: `head -2 results/ASVspoof2019_LA/scores.txt`. Adjust the parser if the format differs.

---

### Task 4: Check 3 — Arena cold-start (manual)

**Goal:** user confirms Arena Space loads with random-baseline row visible.

- [ ] **Step 1: Confirm Arena URL and ask user to open it**

Print the URL: `https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/arena`

Message to user:
> "Open the Arena URL above. Please confirm:
> 1. Page loads without error.
> 2. Overview tab shows one row: random-baseline.
> 3. Per-dataset tab shows ASVspoof2019_LA with EER ≈ 49.87.
> 4. About tab has no warnings listed.
>
> Reply 'pass' or describe what you see."

- [ ] **Step 2: Record user's response**

If PASS: continue.
If FAIL: STOP, debug Arena before continuing. Common causes: Space sleeping (wait 30s and retry), ingest exception (check Space logs), manifest pointing to wrong dataset revision.

---

### Task 5: Check 4 — Edit-and-refresh round-trip

**Goal:** prove Arena reflects a YAML edit after manual Refresh; revert immediately after.

- [ ] **Step 1: Capture current EER value**

The current value is `49.870836165873556`. Target temporary value: `50.870836165873556` (just `+1.00`).

- [ ] **Step 2: Edit the YAML and push to HF**

```bash
cd /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
git pull --rebase
# Edit submissions/random-baseline.yaml: change eer_percent from
#   49.870836165873556  →  50.870836165873556
git diff submissions/random-baseline.yaml   # sanity check: exactly one line changed
git add submissions/random-baseline.yaml
git commit -m "TEMP: Phase 6 smoke test — bump eer +1.00 (will revert)"
git push
```

Use the Edit tool for the YAML change — do NOT use `sed`.

- [ ] **Step 3: Ask user to refresh Arena and confirm**

Message:
> "I pushed a TEMP commit bumping random-baseline's EER from 49.87 to 50.87. Please:
> 1. Open the Arena Space.
> 2. Hit the Refresh button.
> 3. Confirm the EER row now shows ~50.87.
> Reply 'pass' or describe what you see — I will revert immediately after."

- [ ] **Step 4: Revert the commit and push**

```bash
cd /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
git revert --no-edit HEAD
git push
```

- [ ] **Step 5: Ask user to refresh once more and confirm revert**

Message:
> "Reverted. Please Refresh the Arena one more time and confirm the EER is back to 49.87."

PASS if both confirmations succeed. If the bump didn't appear: ingest cache, webhook (shouldn't exist yet), or Refresh button not wired — STOP and debug.

---

### Task 6: Check 5 — Malformed YAML resilience

**Goal:** Arena surfaces a warning for a broken submission file without crashing; remove the bad file after verification.

- [ ] **Step 1: Create malformed YAML and push**

```bash
cd /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
git pull --rebase
```

Use the Write tool to create `submissions/broken-test.yaml` with content:
```
schema_version: 4
system:
  name: broken-test
  description: "[unclosed bracket
scores:
  eer_percent: not-a-number
```
(intentionally invalid: unterminated string + wrong type)

```bash
git add submissions/broken-test.yaml
git commit -m "TEMP: Phase 6 smoke test — malformed YAML (will delete)"
git push
```

- [ ] **Step 2: Ask user to refresh and confirm graceful handling**

Message:
> "I pushed a deliberately-broken submissions/broken-test.yaml. Please:
> 1. Refresh the Arena.
> 2. Confirm: random-baseline row is STILL visible (broken file doesn't break ingest).
> 3. Confirm: About tab lists a warning naming 'broken-test.yaml' (or similar).
> Reply 'pass' or describe what you see — I will delete the bad file after."

- [ ] **Step 3: Delete the broken file and push**

```bash
cd /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
git rm submissions/broken-test.yaml
git commit -m "Remove Phase 6 smoke-test broken YAML"
git push
```

- [ ] **Step 4: Ask user to refresh once more and confirm warning is gone**

Message:
> "Deleted. Please Refresh once more and confirm About tab has no warnings."

PASS if all confirmations succeed. If Arena crashes or the broken file takes down the whole ingest, STOP — fix `arena/ingest.py` error handling before continuing.

---

### Task 7: Report results and update roadmap

**Goal:** mark Phase 6 done in ROADMAP.md only if all five checks passed.

- [ ] **Step 1: Tally results**

Confirm Tasks 1–6 all reported PASS.

- [ ] **Step 2: Update roadmap checklist**

Edit `/home/kirill/speech-spoof-bench/speech-spoof-bench/docs/roadmap/ROADMAP.md` — change the five Phase 6 checklist boxes from `- [ ]` to `- [x]`.

- [ ] **Step 3: Commit**

```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
git add docs/roadmap/ROADMAP.md
git commit -m "Mark Phase 6 smoke test complete"
```

- [ ] **Step 4: Summarize to user**

Print a 3–5 line summary: which checks ran, all PASS, Phase 6 complete, ready for Phase 7.
