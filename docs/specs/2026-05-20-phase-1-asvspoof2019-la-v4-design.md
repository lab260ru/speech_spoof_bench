# Phase 1 — ASVspoof2019_LA in v4 pointer form

**Date:** 2026-05-20
**Scope:** Roadmap Phase 1. Bring the existing `ASVspoof2019_LA` HF dataset repo into conformance with the v4 spec, and relocate the working copy to the larger drive.
**Out of scope:** Phase 2 (pip package), Phase 3 (first submission), and everything downstream.

---

## 1. Motivation

Two unresolved discrepancies in the project right now:

1. **Spec drift.** `docs/roadmap/PLAN.md` is the v3 spec. `docs/roadmap/ROADMAP.md` is the v4 plan and explicitly references "the v4 pointer form" of §1.6. PLAN.md must be promoted to v4 so the dataset repo conforms to one source of truth.
2. **Storage location.** The dataset working copy lives at `/home/kirill/ASVspoof2019_LA/` (13 GB on the home filesystem). It belongs on `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/` — the same drive used for the other anti-spoofing benchmarks.

Phase 1 closes both.

## 2. Why v4 changes §1.6

The v3 submission format stores `scores/<slug>.txt` inside the dataset repo. With every submission, the dataset repo grows; cloning it (a basic operation for any submitter or arena worker) becomes proportionally slower. v4 inverts the ownership: the scores file lives in the submitter's own model repo, and the submission YAML carries a pinned, commit-immutable `scores_url` plus `scores_sha256`. The dataset repo only ever stores small YAML files.

Secondary benefits:
- Forces every submission to be backed by a publicly addressable, sha-verifiable artifact under the submitter's control.
- `nightly-revalidate` (Phase 8f) can re-check `scores_url` reachability and sha drift without doing any local IO.
- The HF CLI submitter flow becomes two `huggingface-cli upload` calls — no git clone of the dataset.

## 3. Current state

| Item | Status |
|---|---|
| Repo on HF | `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA`, exists. |
| Local working copy | `/home/kirill/ASVspoof2019_LA/`, 13 GB, branch `master`, head `06296dc`, clean. |
| Destination drive | `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/`, empty. |
| Parquet schema | `{path, audio, label, notes}` — already v4-compliant (commit `797cd9d`). |
| README frontmatter | `arena-ready` tag + `arxiv:` list present — compliant. |
| `eval.yaml` | Matches §1.5 — compliant. |
| Citation block | Present in README — compliant. |
| `submissions/scores/` | Exists with `.gitkeep` — must be deleted (v4 has no in-repo scores). |
| `submissions/README.md` | v3 prose with `scores/<slug>.txt` instructions — must be rewritten. |
| `submissions/results_template.yaml` | v3 schema with `artifact.scores_file` — must be rewritten for v4. |

## 4. v4 submission YAML schema (§1.6 replacement)

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
  revision: 7f3a9b1c          # commit sha of the dataset at scoring time
  split: test

scores:
  eer_percent: 0.83
  n_trials: 71237
  n_skipped: 0

artifact:
  # Pinned, commit-immutable URL. Pattern:
  #   https://huggingface.co/<owner>/<repo>/resolve/<commit-sha>/.eval_results/<dataset-org>/<dataset-name>/scores.txt
  scores_url: https://huggingface.co/<owner>/<repo>/resolve/<sha>/.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt
  scores_sha256: 4f9b...
  bench_version: speech-spoof-bench==0.3.1

reproduction:
  reproduced_by: SpeechAntiSpoofingBenchmarks
  reproduced_at: 2026-05-20
  reproduced_bench_version: speech-spoof-bench==0.3.1
  match: scoring              # "scoring" | "inference"

submitter:
  hf_username: kborodin
  contact: k.n.borodin@mtuci.ru

submitted_at: 2026-05-19
notes: "..."
```

### Diff vs. v3

- `schema_version: 1` → `4`.
- `artifact.scores_file` (relative path inside dataset repo) → `artifact.scores_url` (pinned absolute HF URL).
- `system.paper` is now a structured block (`arxiv_id` + `url` + `bibtex`), not a bare URL string.
- `submissions/scores/` is removed from §1.1 layout.

### Rules (prose for §1.6)

- `scores_url` MUST be pinned by commit sha. URLs with `/resolve/main/` or any branch ref are invalid — they're mutable and break sha verification.
- `scores_sha256` MUST be sha256 of the file at `scores_url` at submission time. `speech-spoof-bench reproduce --scoring` (Phase 7) fetches the URL and verifies.
- If the model repo at `scores_url` is deleted or rewritten, the submission becomes irreproducible and is auto-flagged by `nightly-revalidate` (Phase 8f).
- The maintainer fills in the `reproduction:` block at merge time; submitters leave it empty.

## 5. Submitter workflow (rewritten `submissions/README.md`)

1. Run the bench locally:
   ```bash
   speech-spoof-bench run \
     --model-module <pkg>:<Class> \
     --datasets SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
   ```
   → produces `results/ASVspoof2019_LA/scores.txt`.

2. Upload `scores.txt` to your HF model repo:
   ```bash
   huggingface-cli upload <your-owner>/<your-repo> \
     results/ASVspoof2019_LA/scores.txt \
     .eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt \
     --repo-type=model
   ```
   Note the returned commit sha.

3. Build the pinned `scores_url`, compute `sha256sum scores.txt`, copy `results_template.yaml` to `<your-slug>.yaml`, fill in.

4. Open the PR on the dataset repo:
   ```bash
   huggingface-cli upload \
     SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA \
     <your-slug>.yaml submissions/<your-slug>.yaml \
     --repo-type=dataset \
     --create-pr \
     --commit-message="Add <your-slug> submission"
   ```

5. Maintainer runs `speech-spoof-bench reproduce --scoring <PR-branch>`, fills in `reproduction:`, merges.

Phase 7's `speech-spoof-bench submit` will eventually wrap steps 1–4 into a single CLI call.

## 6. Execution plan

Execute in order. Each step is verifiable; do not advance until the prior step is verified.

### Step 1 — Move the dataset working copy

1. `mkdir -p /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks` (already exists; idempotent).
2. `rsync -aHAX --info=progress2 /home/kirill/ASVspoof2019_LA/ /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/`
3. Verify:
   - `git -C <new> fsck --full` exits 0.
   - `git -C <new> status` reports clean.
   - `git -C <new> log -1 --format=%h` starts with `06296dc` (matches the source head).
   - `rsync -nai --checksum /home/kirill/ASVspoof2019_LA/ <new>/` reports no diffs.
4. Only after all four verifications pass: `rm -rf /home/kirill/ASVspoof2019_LA`.

### Step 2 — Promote PLAN.md to v4

Edits inside `/home/kirill/speech-spoof-bench/speech-spoof-bench/docs/roadmap/PLAN.md`:

- Title: `... Full Infrastructure Spec v3` → `v4`.
- §1.1 layout block: remove the `submissions/scores/<slug>.txt` line.
- §1.6: replace the entire YAML block and surrounding prose with §4 of this design doc.
- §1.9 validation list: update the "scores SHA matches the file" bullet to "scores SHA matches the file fetched from `scores_url`".
- §1.10 DoD: no change (already references `reproduction:`).

### Step 3 — Update ROADMAP.md workspace layout

In `docs/roadmap/ROADMAP.md`, the "Workspace layout" paragraph (lines ~13, 28):

- Replace `/home/kirill/datasets/` with `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/`.
- Tick the Phase 1 checklist boxes as they're completed in Step 4.

Commit steps 2 + 3 together:

```
docs: promote spec to v4 (pointer-style submissions, new dataset path)
```

### Step 4 — Dataset repo conformance

Run inside the new dataset working copy:

1. `git rm -r submissions/scores/`.
2. Overwrite `submissions/README.md` with the §5 content of this design doc.
3. Overwrite `submissions/results_template.yaml` with the §4 schema (all fields present, values empty/placeholder).
4. Commit:
   ```
   feat: migrate submission format to v4 pointer style

   - Remove submissions/scores/ (scores now live in submitter model repos)
   - Rewrite submissions/README.md with HF CLI submitter flow
   - Rewrite submissions/results_template.yaml with v4 schema
   ```

### Step 5 — Push

From the new location:
```bash
git push origin master
```

Confirm:
- HF web UI shows updated `submissions/README.md`.
- `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/tree/main/submissions/` has no `scores/` folder.
- `from datasets import load_dataset; load_dataset("SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA", split="test")` still works.

## 7. Definition of done

Per roadmap Phase 1:

- [ ] `/home/kirill/ASVspoof2019_LA` no longer exists; the only working copy is at the new path.
- [ ] HF dataset page shows v4 layout (no `submissions/scores/`, v4 README, v4 template).
- [ ] README frontmatter validates (`arena-ready`, `arxiv:` list present).
- [ ] `load_dataset(..., split="test")` works.
- [ ] PLAN.md is the v4 spec; ROADMAP.md references match.
- [ ] PLAN.md and ROADMAP.md committed and pushed to the project's GitHub repo.

## 8. Deferred to later phases

- JSON Schema for the v4 YAML — Phase 7 (`validate`).
- `speech-spoof-bench reproduce --scoring` implementation — Phase 7.
- First actual `<slug>.yaml` submission (random baseline) — Phase 3, depends on Phase 2 pip package.
- Nightly URL-reachability checks — Phase 8f.

## 9. Risks

- **rsync of 13 GB across drives is slow.** Estimated 10–30 min on a typical USB/SATA bridge. Mitigation: run in foreground with `--info=progress2`; don't background it.
- **Git LFS (if any) might not survive rsync.** The repo doesn't appear to use LFS (parquet shards are in the working tree, `.gitattributes` is small). Verified by `git fsck`.
- **HF push could be rejected for any frontmatter regression.** Mitigation: explicit audit pass before push.
- **PLAN.md changes invalidate prose elsewhere in PLAN.md that still references v3 scores file.** Mitigation: grep for `scores_file`, `scores/`, `submissions/scores` across PLAN.md before committing Step 2.
