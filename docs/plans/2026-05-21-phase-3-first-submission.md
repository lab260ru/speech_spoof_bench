# Phase 3 — First Manual Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one valid v4 submission YAML in the `ASVspoof2019_LA` dataset repo, pointing at a `scores.txt` hosted in a model repo, with the `reproduction:` block filled in by the maintainer.

**Architecture:** Procedural one-shot. Re-run the seeded random baseline, verify byte-identical scores, upload to `SpeechAntiSpoofingBenchmarks/random-baseline-asas`, capture the commit sha, hand-author `submissions/random-baseline.yaml` in the LA working copy, push.

**Tech Stack:** `huggingface_hub` (Python), `git`, `bash`, the local `speech-spoof-bench` CLI built in Phase 2.

**Spec:** `docs/specs/2026-05-21-phase-3-first-submission-design.md`

---

## File Structure

This phase creates exactly one new file and modifies none:

- **Create**: `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/submissions/random-baseline.yaml` — the v4 pointer-style submission for the random baseline.

Two HF-side artifacts are also produced (not in the project repo):
- `SpeechAntiSpoofingBenchmarks/random-baseline-asas/.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt`
- `SpeechAntiSpoofingBenchmarks/random-baseline-asas/README.md` (minimal stub)

A throwaway script lives at `/tmp/phase3_upload.py` during execution; not committed.

---

## Task 1: Verify scores.txt determinism

**Files:**
- Read: `/home/kirill/speech-spoof-bench/speech-spoof-bench/results/ASVspoof2019_LA/scores.txt` (existing)
- Read: `/home/kirill/speech-spoof-bench/speech-spoof-bench/src/speech_spoof_bench/examples/random_baseline.py` (already `seed=0`)

- [ ] **Step 1: Sanity-check the existing scores.txt sha256**

Run:
```bash
sha256sum /home/kirill/speech-spoof-bench/speech-spoof-bench/results/ASVspoof2019_LA/scores.txt
```
Expected: `71ac000c0712a4551873dba87183e746cb9730cd5ab17aaa87892009bde55587  ...`

If it doesn't match the sha256 recorded in `results/ASVspoof2019_LA/result.yaml`, stop and investigate — the file has changed since Phase 2.

- [ ] **Step 2: Re-run the random baseline into a fresh output directory**

Run:
```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
speech-spoof-bench run \
    --model-module speech_spoof_bench.examples.random_baseline:RandomBaseline \
    --datasets SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA \
    --output-dir /tmp/phase3_rerun
```
Expected: completes, produces `/tmp/phase3_rerun/ASVspoof2019_LA/scores.txt` + `result.yaml`.

If the CLI does not accept `--output-dir`, check `speech-spoof-bench run --help` and adapt. Do not modify the CLI in this phase.

- [ ] **Step 3: Diff the new scores.txt against the Phase 2 file**

Run:
```bash
diff -q /home/kirill/speech-spoof-bench/speech-spoof-bench/results/ASVspoof2019_LA/scores.txt \
        /tmp/phase3_rerun/ASVspoof2019_LA/scores.txt
sha256sum /tmp/phase3_rerun/ASVspoof2019_LA/scores.txt
```
Expected:
- `diff -q` prints nothing (files identical).
- sha256 equals `71ac000c0712a4551873dba87183e746cb9730cd5ab17aaa87892009bde55587`.

**If the diff is non-empty:** STOP. The runner's row iteration order is not stable — that's a Phase 2 bug. File an issue and do not proceed.

- [ ] **Step 4: Clean up the re-run output**

Run:
```bash
rm -rf /tmp/phase3_rerun
```

- [ ] **Step 5: No commit** (no code changed in this task).

---

## Task 2: Upload scores.txt to the model repo

**Files:**
- Create (HF side): `SpeechAntiSpoofingBenchmarks/random-baseline-asas/README.md`
- Create (HF side): `SpeechAntiSpoofingBenchmarks/random-baseline-asas/.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt`
- Create (temp): `/tmp/phase3_upload.py`

- [ ] **Step 1: Confirm HF auth is active**

Run:
```bash
huggingface-cli whoami
```
Expected: prints a username (any HF account with write access to `SpeechAntiSpoofingBenchmarks`). If it errors with "Not logged in", stop and ask the user to run `huggingface-cli login`.

- [ ] **Step 2: Write the upload script**

Create `/tmp/phase3_upload.py`:
```python
"""One-shot upload of scores.txt + minimal README to the random-baseline model repo.

Captures the commit OID of the scores.txt upload, which becomes <UPLOAD_SHA>
in the dataset submission YAML.
"""
from huggingface_hub import HfApi

REPO = "SpeechAntiSpoofingBenchmarks/random-baseline-asas"
LOCAL_SCORES = "/home/kirill/speech-spoof-bench/speech-spoof-bench/results/ASVspoof2019_LA/scores.txt"
REMOTE_SCORES = ".eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt"

README = """\
# random-baseline-asas

Reference random baseline for the SpeechAntiSpoofingBenchmarks arena.

Returns `N(0, 1)` for every utterance using a fixed seed (`seed=0`).
EER ≈ 50% by construction. No checkpoint — the model is the seed.

See evaluation results under `.eval_results/<dataset-org>/<dataset-name>/scores.txt`.

- Code: https://github.com/SpeechAntiSpoofingBenchmarks/speech-spoof-bench
- Arena: https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/arena
"""

api = HfApi()

# 1. README first (low-stakes; warms the repo if it was empty).
readme_info = api.upload_file(
    path_or_fileobj=README.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=REPO,
    repo_type="model",
    commit_message="docs: add README for random-baseline-asas",
)
print(f"README commit: {readme_info.oid}")

# 2. scores.txt — capture its commit OID; this is what we pin in the submission YAML.
scores_info = api.upload_file(
    path_or_fileobj=LOCAL_SCORES,
    path_in_repo=REMOTE_SCORES,
    repo_id=REPO,
    repo_type="model",
    commit_message="feat: upload scores.txt for ASVspoof2019_LA (Phase 3)",
)
print(f"SCORES_COMMIT_OID={scores_info.oid}")
print(f"SCORES_URL=https://huggingface.co/{REPO}/resolve/{scores_info.oid}/{REMOTE_SCORES}")
```

- [ ] **Step 3: Run the upload script**

Run:
```bash
python /tmp/phase3_upload.py
```
Expected output (last two lines are the values you'll paste into the YAML in Task 3):
```
README commit: <some sha>
SCORES_COMMIT_OID=<40-char sha>
SCORES_URL=https://huggingface.co/SpeechAntiSpoofingBenchmarks/random-baseline-asas/resolve/<40-char sha>/.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt
```

**Save** the `SCORES_COMMIT_OID` value — it is required in Task 3, Step 2.

- [ ] **Step 4: Verify the pinned URL is reachable and the sha matches**

Substitute the captured OID into the URL, then:
```bash
curl -sSL -o /tmp/phase3_fetched.txt \
  "https://huggingface.co/SpeechAntiSpoofingBenchmarks/random-baseline-asas/resolve/<SCORES_COMMIT_OID>/.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt"
sha256sum /tmp/phase3_fetched.txt
```
Expected sha256: `71ac000c0712a4551873dba87183e746cb9730cd5ab17aaa87892009bde55587`.

If the sha differs, STOP — either the upload corrupted the file or HF served a transformed version (LFS pointer, etc.). Investigate before proceeding.

- [ ] **Step 5: Clean up**

Run:
```bash
rm /tmp/phase3_fetched.txt /tmp/phase3_upload.py
```

- [ ] **Step 6: No git commit** (no project-repo files changed).

---

## Task 3: Author submissions/random-baseline.yaml

**Files:**
- Create: `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/submissions/random-baseline.yaml`

- [ ] **Step 1: Confirm LA dataset HEAD short sha**

Run:
```bash
cd /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
git rev-parse --short=8 HEAD
git status
```
Expected:
- `git rev-parse` prints `151aa4c6` (or whatever HEAD now is — record this value).
- `git status` shows a clean working tree on `main`.

If HEAD has moved beyond `151aa4c6`, use whatever the new short sha is — it just needs to be the sha at the moment of submission.

- [ ] **Step 2: Create the submission YAML**

Substitute `<UPLOAD_SHA>` with the `SCORES_COMMIT_OID` from Task 2, Step 3, and `<LA_REVISION>` with the short sha from Step 1 above.

Create `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/submissions/random-baseline.yaml`:

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
  revision: <LA_REVISION>
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

- [ ] **Step 3: YAML parse check**

Run:
```bash
python -c "import yaml, sys; yaml.safe_load(open('/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/submissions/random-baseline.yaml')); print('ok')"
```
Expected: prints `ok`. If `yaml.YAMLError`, fix the YAML.

- [ ] **Step 4: Spot-check key fields**

Run:
```bash
python <<'PY'
import yaml
with open("/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/submissions/random-baseline.yaml") as f:
    d = yaml.safe_load(f)
assert d["schema_version"] == 4
assert d["system"]["slug"] == "random-baseline"
assert d["dataset"]["id"] == "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA"
assert d["dataset"]["revision"] and "<" not in d["dataset"]["revision"]
assert d["artifact"]["scores_url"].startswith("https://huggingface.co/")
assert "/resolve/" in d["artifact"]["scores_url"]
assert "<UPLOAD_SHA>" not in d["artifact"]["scores_url"]
assert d["artifact"]["scores_sha256"] == "71ac000c0712a4551873dba87183e746cb9730cd5ab17aaa87892009bde55587"
assert d["reproduction"]["match"] == "scoring"
print("ok")
PY
```
Expected: prints `ok`. Any `AssertionError` means a substitution was missed — fix and re-run.

- [ ] **Step 5: Cross-check sha by re-fetching from the pinned URL**

Run:
```bash
URL=$(python -c "import yaml; print(yaml.safe_load(open('/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/submissions/random-baseline.yaml'))['artifact']['scores_url'])")
curl -sSL "$URL" | sha256sum
```
Expected: `71ac000c0712a4551873dba87183e746cb9730cd5ab17aaa87892009bde55587  -`.

If the sha differs, the URL in the YAML is wrong (most likely `<UPLOAD_SHA>` wasn't substituted, or the wrong sha was pasted). Fix before proceeding.

- [ ] **Step 6: No commit yet** — Task 4 stages and commits.

---

## Task 4: Push to the LA dataset repo

**Files:**
- Commit (LA repo): `submissions/random-baseline.yaml`

- [ ] **Step 1: Confirm there are no other modifications in the LA working copy**

Run:
```bash
cd /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
git status
```
Expected: only `submissions/random-baseline.yaml` is shown as new. If anything else is dirty, ask the user before committing.

- [ ] **Step 2: Stage the file**

Run:
```bash
cd /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
git add submissions/random-baseline.yaml
```

- [ ] **Step 3: Commit**

Run:
```bash
cd /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
git commit -m "feat: add random baseline submission (Phase 3)"
```
Expected: one new commit on `main`.

- [ ] **Step 4: Push to HF**

Run:
```bash
cd /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
git push origin main
```
Expected: push succeeds.

If push fails on auth, ask the user to confirm their HF git credentials (token in `~/.netrc` or `huggingface-cli login`). Do not bypass.

- [ ] **Step 5: Verify the file is reachable on HF**

Run:
```bash
curl -sSI "https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/resolve/main/submissions/random-baseline.yaml" | head -1
```
Expected: `HTTP/2 200` (or 302 → 200).

---

## Task 5: End-to-end verification

**Files:** none modified.

- [ ] **Step 1: Re-fetch the submission YAML from HF and the scores file via its pinned URL**

Run:
```bash
python <<'PY'
import urllib.request, yaml, hashlib

YAML_URL = "https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/resolve/main/submissions/random-baseline.yaml"

with urllib.request.urlopen(YAML_URL) as r:
    sub = yaml.safe_load(r.read())

scores_url = sub["artifact"]["scores_url"]
expected_sha = sub["artifact"]["scores_sha256"]

with urllib.request.urlopen(scores_url) as r:
    body = r.read()

actual_sha = hashlib.sha256(body).hexdigest()
assert actual_sha == expected_sha, f"sha mismatch: {actual_sha} vs {expected_sha}"

# Sanity: scores.txt has one line per utterance, n_trials matches
lines = [l for l in body.decode().splitlines() if l.strip()]
assert len(lines) == sub["scores"]["n_trials"], f"n_trials mismatch: {len(lines)} vs {sub['scores']['n_trials']}"

print("ok — YAML reachable, sha matches, n_trials matches")
PY
```
Expected: prints the `ok` line. Any assertion failure means the submission is broken.

- [ ] **Step 2: Update the roadmap checklist**

Edit `/home/kirill/speech-spoof-bench/docs/roadmap/ROADMAP.md` Phase 3 section: tick the five Phase 3 checkboxes (`- [ ]` → `- [x]`).

- [ ] **Step 3: Commit the roadmap update in the project repo**

Run:
```bash
cd /home/kirill/speech-spoof-bench
git add docs/roadmap/ROADMAP.md
git commit -m "docs: mark Phase 3 complete in roadmap"
```

---

## Done when

- `submissions/random-baseline.yaml` is live on `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA`.
- The `scores_url` in that YAML resolves (HTTP 200), and its sha256 matches `scores_sha256`.
- `n_trials` in the YAML equals the line count of the fetched scores file.
- Phase 3 checkboxes in `ROADMAP.md` are ticked and committed.
