# SpeechAntiSpoofingBenchmarks — Implementation Roadmap

Derived from `SpeechAntiSpoofingBenchmarks_Plan.md` (v4). The order is:

> **1 dataset → pip package skeleton → 1 model (random baseline) → manifest → Arena MVP → end-to-end smoke test → CI/CD layer → badge layer → scale (more datasets, more models)**

The first six phases prove the data path end-to-end with the cheapest possible Arena. Only after that do we add the CI/webhook/badge machinery — they make a working system *automatic*, but they're not on the critical path to "does this thing work at all?"

---

## Workspace layout

All repo working copies live directly under `/home/kirill/speech-spoof-bench/`. Dataset working copies (parquet, audio, git history) live on the larger drive at `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/<dataset>/` and push directly to HF — they're not tracked in the project repo.

```
speech-spoof-bench/                     # this folder (the project root)
├── ROADMAP.md                          # this file
├── SpeechAntiSpoofingBenchmarks_Plan.md # v4 spec
├── speech-spoof-bench/                 # pip package + CI workflows (→ GitHub)
├── arena/                              # Docker Space (→ HF Space)
├── arena-manifest/                     # one-file manifest repo (→ HF dataset repo)
└── dataset-builders/                   # build scripts only, no data
    └── ASVspoof2019_LA/
        ├── build_parquet.py
        └── README.md
```

Datasets-as-files live at `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/<dataset>/` (one git working copy per HF dataset) or just on HF. This folder holds **only code and config**.

---

## Phase 0 — Workspace setup

**Goal**: clean slate to start working in.

- [x] `git init` each of the four sub-repos at the project root with appropriate `.gitignore`.
- [x] Create empty placeholder READMEs so each repo has a root commit.
- [x] Decide HF org name (assume `SpeechAntiSpoofingBenchmarks` per spec).
- [x] Create org on HF if not already (`huggingface.co/organizations`).
- [x] Create empty target HF repos: `arena-manifest` (dataset), `arena` (Space), `ASVspoof2019_LA` (dataset — already exists per user).
- [x] Create empty GitHub repo `SpeechAntiSpoofingBenchmarks/speech-spoof-bench`.

**Done when**: all five remote repos exist and four local working copies are git-initialized.

---

## Phase 1 — Single dataset (ASVspoof2019_LA, v4 pointer form)

**Goal**: dataset repo matches the v4 spec exactly. No submissions yet. No CI yet.

- [x] Audit current `ASVspoof2019_LA` repo against §1.1 layout. Remove any `submissions/scores/` directory if present (v4 deletes it).
- [x] Verify schema is `{path, audio, label, notes}` per §1.2.
- [x] Verify README has frontmatter per §1.4: `tags` includes `arena-ready`, `arxiv:` list present, license correct.
- [x] Verify `eval.yaml` matches §1.5 (inspect-ai shape, `metrics: [eer_percent]`).
- [x] Add/refresh citation block in README body (arXiv link + BibTeX).
- [x] Create `submissions/README.md` and `submissions/results_template.yaml` with the new pointer-style template (§1.6). Empty `submissions/<slug>.yaml` set for now.
- [x] Commit + push.

**Done when**: visiting `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA` shows the v4 layout, frontmatter is valid, `load_dataset(..., split="test")` works.

---

## Phase 2 — Pip package skeleton (just enough for the random baseline)

**Goal**: `pip install -e .` works; you can run a random baseline against the LA dataset and produce a `scores.txt`.

In `./speech-spoof-bench/`:

- [x] `pyproject.toml` (core deps only: `datasets`, `huggingface_hub`, `numpy`, `pyyaml`, `jsonschema`, `scipy`).
- [x] `src/speech_spoof_bench/model.py` — `AntiSpoofingModel` ABC (§2.2).
- [x] `src/speech_spoof_bench/metrics/eer.py` + `__init__.py` with `register_metric` decorator + `MetricResult` (§2.6).
- [x] `src/speech_spoof_bench/loader.py` — local-or-HF dispatch reading `eval.yaml`.
- [x] `src/speech_spoof_bench/runner.py` — streaming row iteration, calls `model.score_batch`, writes `scores.txt`. Per-item fallback on batch errors.
- [x] `src/speech_spoof_bench/benchmark.py` — `Benchmark.run` orchestrator (§2.3). `load`/`unload` once per evaluation in `try/finally`.
- [x] `src/speech_spoof_bench/cache.py` — `cleanup=True` purges HF cache for a given dataset id.
- [x] `src/speech_spoof_bench/manifest.py` — stub (full impl phase 4).
- [x] `src/speech_spoof_bench/examples/random_baseline.py` — concrete `AntiSpoofingModel` returning `np.random.randn()`.
- [x] `src/speech_spoof_bench/cli.py` — at minimum: `run`, `list`, `validate-dataset` (latter is a stub that confirms the dataset loads).
- [x] One unit test per metric in `tests/metrics/test_eer.py` (synthetic scores → known EER). Plus tests for loader, runner, benchmark, model, cli.

**Skip for now**: `submit`, `ci verify-pr`, `scaffold-dataset`, webhook handler, full `validate-dataset` checks, `reproduce`. Those come in Phase 6+.

**Done when**:
```bash
speech-spoof-bench run \
    --model-module speech_spoof_bench.examples.random_baseline:RandomBaseline \
    --datasets SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
```
produces a `results/ASVspoof2019_LA/scores.txt` + `result.yaml` locally. EER is around 50%.

---

## Phase 3 — First submission, manually authored

**Goal**: produce a working `submissions/<slug>.yaml` in the dataset repo, **manually** (no `submit` CLI yet). Proves the v4 pointer-style schema works end-to-end before we automate it.

- [x] Create a personal HF model repo `<you>/random-baseline-asas` (just a stub, no checkpoint).
- [x] Upload `scores.txt` produced in Phase 2 to `<you>/random-baseline-asas/.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt`. Note the commit sha.
- [x] Compute `scores_sha256` locally (`sha256sum scores.txt`).
- [x] Hand-author `submissions/random-baseline.yaml` in the LA dataset repo per §1.6: pointer URL + sha + paper (use a placeholder arXiv ID and a real BibTeX entry) + `reproduction:` block filled in (level: `scoring`, `reproduced_at: <today>`).
- [x] For the random baseline, paper field can point to ASVspoof2019 paper itself (the dataset paper, since "random" has no paper). Document this convention.
- [x] Push directly to main (you're the maintainer, no PR/CI yet).

**Done when**: one valid v4 submission YAML exists in the dataset repo, references real `scores.txt` in a model repo, sha256 matches.

---

## Phase 4 — Manifest

**Goal**: the one-file manifest the Arena and pip package read.

In `./arena-manifest/`:

- [x] `manifest.yaml` per §4 — `ranking_version: v1`, `tiers` (gold/silver/bronze), `core_set` with just LA pinned to current commit sha, empty `extended`.
- [x] Push to `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/arena-manifest`.

**Done when**: `speech-spoof-bench manifest` CLI (add this to Phase 2 if not yet) prints the manifest contents. ✓

---

## Phase 5 — Arena MVP (read-only, no webhook)

**Goal**: a working leaderboard Space showing one row (random baseline on LA). No webhooks, no `cache.json`, no Docker — just Gradio + cold-start fetch. The simplest thing that displays the data.

In `./arena/`:

- [x] `app.py` — Gradio app with 3 tabs: Overview, Per-dataset, About.
- [x] `ingest.py` — at cold start: fetch manifest → for each dataset, `list_repo_files` → fetch each `submissions/*.yaml` → build in-memory `Row` list (§3.4).
- [x] `ranking.py` — minimal tier assignment (§3.7).
- [x] `ui/overview.py`, `ui/per_dataset.py`, `ui/about.py` — Gradio DataFrames.
- [x] Manual "Refresh" button that re-runs `ingest.py`.
- [x] Push to `huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/arena` as a plain Gradio Space (not Docker yet — Docker comes in Phase 7).

**Skip for now**: system-detail page, paper rendering, submit tab, Docker, webhook, `cache.json`.

**Done when**: opening the Space URL shows the random baseline row on LA with its EER. Refresh button works.

---

## Phase 6 — End-to-end smoke test ✅

**Stop here and verify everything works** before adding automation.

Checklist:
- [x] `speech-spoof-bench run` on LA produces `scores.txt` matching what's in the model repo.
- [x] EER computed locally matches `scores.eer_percent` in the submission YAML.
- [x] Arena cold-starts and shows the row.
- [x] Manually edit the submission YAML (e.g. change the EER value), push, hit Refresh in Arena — the change appears.
- [x] Manually break the YAML (malformed) — Arena logs a warning in About, doesn't crash.

If any of these fail: fix before proceeding. Do not add more datasets, models, or automation until the smoke test is green.

---

## Phase 7a — Validators (offline + maintainer gate)

**Goal**: maintainer-side and submitter-side validation tools. No HF write
operations. Provides the gate the spec marks mandatory before merging any
submission (§1.7).

Spec: `docs/specs/2026-05-22-phase-7a-validators-design.md`.

- [x] `validate-submission <yaml>` — standalone schema check, no network (§1.6 / §2.5).
- [x] `validate-dataset <repo-or-path> [--skip-submissions]` — full §1.9 checks: schema, sample rate, uniqueness, README frontmatter, eval.yaml shape + metric registry, plus per-submission schema + `scores_url` reachability + sha verification. Aggregating report.
- [x] `reproduce --scoring <yaml> [--tolerance]` — fetches `scores_url`, verifies sha, streams labels from pinned dataset revision (no audio decode via `select_columns`), recomputes every registered metric, diffs against claimed values.
- [x] `reproduce --inference` wired as `NotImplementedError` (lands in Phase 8).

**Done when**: `reproduce --scoring submissions/random-baseline.yaml` on the live LA submission exits 0; `validate-dataset SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA` exits 0 with all checks green.

**Deferred** (do later, not blocking):
- `reproduce --scoring <repo>@<ref>:<file>` — fetch a submission YAML directly from an HF dataset PR branch instead of requiring a local checkout.

---

## Phase 7b — Authoring (`submit`, `scaffold-dataset`)

**Goal**: humans no longer hand-author submission YAMLs.

Spec: TBD (own brainstorming cycle once 7a lands).

- [ ] `submit` (§2.5a) — one command: run + upload scores to model repo + build YAML + open HF PR on dataset repo.
- [ ] `scaffold-dataset` (§2.5, §3.8 step 1) — produces the skeleton for a new dataset repo.

**Done when**: `speech-spoof-bench submit --model-module speech_spoof_bench.examples.random_baseline:RandomBaseline --model-repo <you>/random-baseline-asas` opens a PR on LA that contains a valid v4 submission YAML. Manual merge → Arena reflects it on next Refresh.

---

## Phase 8 — CI/CD layer

**Goal**: PRs validated automatically; comments posted back to HF discussions; merging triggers Arena refresh without manual Refresh.

- [ ] **8a. Upgrade Arena to Docker Space.** Move `app.py` + new `webhook.py` into a `Dockerfile`-built image. Mount Gradio at `/` and FastAPI at `/webhook`. Verify Arena still works identically.
- [ ] **8b. Add `cache.json` + committed persistence.** Cold start reads `cache.json` first; webhook handler updates it and commits back using `SPACE_COMMIT_TOKEN`. Cold-start time drops to <3s.
- [ ] **8c. Configure HF webhooks.** Subscribe Arena's `/webhook` to `repo.content` events on the LA dataset repo. Test with a dummy commit — Arena refreshes within 60s.
- [ ] **8d. GitHub Actions: `verify-hf-pr.yml`.** Triggered by `workflow_dispatch`. Calls `speech-spoof-bench ci verify-pr`. Posts CI comment to HF discussion via `HF_BOT_TOKEN`. Test on a dummy PR.
- [ ] **8e. Webhook → Actions bridge.** When `/webhook` receives a PR-branch event, it calls `gh workflow run verify-hf-pr.yml` using `GH_PAT`. End-to-end: open HF PR → CI comment appears within 2 minutes.
- [ ] **8f. `nightly-revalidate.yml`.** Cron job. Walks all merged submissions; opens issue on 404 / sha mismatch / EER drift.
- [ ] **8g. Secrets audit.** Confirm all four secrets from §3.5.5 are minimum-scope.

**Done when**: opening an HF PR on LA → CI comment appears → maintainer merges → Arena reflects within 60s. All automatic.

---

## Phase 9 — Badge layer

**Goal**: post-merge CI comment generates a paste-ready `.eval_results/<dataset>/result.yaml`; submitter pastes it; HF model page shows the leaderboard badge.

- [ ] `src/speech_spoof_bench/badge.py` — generates `result.yaml` per §3.6.2.
- [ ] `result.schema.json` — JSON Schema validator.
- [ ] `ci verify-pr` post-merge step — emits a second comment with the snippet and one-liner upload command.
- [ ] Manually verify once: paste into random-baseline model repo → badge renders on `huggingface.co/<you>/random-baseline-asas`.

**Done when**: a backlink badge from the random baseline's model page to the Arena is live.

---

## Phase 10 — Arena polish

Now that the system works, fill in the rest of §3.2:

- [ ] System-detail page with paper rendering, BibTeX copy button, cached arXiv abstract.
- [ ] Submit tab with the exact command + walkthrough.
- [ ] Per-dataset paper column with arXiv badge.
- [ ] About tab fully populated (subscribed datasets, last-refreshed per dataset, link to CI).
- [ ] 📄 paper badge surfaced on rows.

---

## Phase 11 — Scale: second dataset

**Goal**: prove the schema generalizes. Use `scaffold-dataset` end-to-end.

- [ ] Pick `InTheWild` (smallest of the planned next ones).
- [ ] `speech-spoof-bench scaffold-dataset --name InTheWild --source-paper https://arxiv.org/abs/2203.16263 --output-dir ./dataset-builders/InTheWild`.
- [ ] Build parquet, validate locally.
- [ ] Push to HF.
- [ ] Add to manifest (a PR on `arena-manifest`).
- [ ] Verify Arena auto-subscribes to its webhook on next restart and the new dataset appears in the dropdown.

**Stop-the-line**: if the schema needs to bend to fit InTheWild, the schema is wrong — revisit §1.2 / §1.6 before continuing.

---

## Phase 12 — Scale: second model

- [ ] Implement a real model wrapper (e.g. AASIST) as `examples/aasist.py`.
- [ ] Run `speech-spoof-bench submit --model-module speech_spoof_bench.examples.aasist:AASIST --datasets all --model-repo <you>/aasist-asas`.
- [ ] CI verifies on both datasets; maintainer merges; Arena reflects both rows.
- [ ] AASIST EER on LA must match published baseline within 0.05% — if not, debug.

---

## Phase 13 — Scale onward

Per §7 build order: add `ASVspoof2021_LA`, `ASVspoof2021_DF`, then `ASVspoof2019_PA`, `WaveFake`, `ASVspoof5`. Add more model baselines as community contributions roll in.

When ready:
- [ ] Apply for HF Benchmarks beta allow-listing via `OpenEvals/README` discussion.
- [ ] Once audio support lands in inspect-ai `field_spec`, wire up `--via-hf-jobs` and the `verified` badge.
- [ ] Apply for HF Jobs compute grant to scale `★ inference` verification.

---

## Critical-path summary

| # | What | Output | Approx. effort |
|---|---|---|---|
| 0 | Workspace setup | 5 repos exist | 1 day |
| 1 | LA dataset in v4 form | Working dataset repo | 1–2 days |
| 2 | Pip package skeleton | `run` works locally | 3–5 days |
| 3 | One manual submission | YAML on HF | 1 day |
| 4 | Manifest | YAML on HF | 0.5 day |
| 5 | Arena MVP (no webhook) | Live Space | 3–4 days |
| **6** | **Smoke test ✅** | **Verified end-to-end** | **0.5 day** |
| 7a | Validators (`validate-submission`, `validate-dataset`, `reproduce --scoring`) | Maintainer gate | 2–3 days |
| 7b | Authoring (`submit`, `scaffold-dataset`) | CLI complete | 2–3 days |
| 8 | CI/CD layer | Auto-validating PRs | 4–6 days |
| 9 | Badge layer | Backlink badges live | 1–2 days |
| 10 | Arena polish | Full UX | 2–3 days |
| 11 | Second dataset | Schema generalizes | 2–4 days |
| 12 | Second model | AASIST baseline | 2–3 days |
| 13+ | Scale | Full Core Set | ongoing |

Total to first usable system (Phase 6): **~2 weeks**. Total to fully automated (Phase 9): **~4–5 weeks**.

---

## What NOT to do early

- Don't build Docker / webhooks / CI before Phase 6. They obscure failures in the data path.
- Don't `submit`-automate before Phase 3 hand-authoring works — you'll be debugging the wrong layer.
- Don't add a second dataset before Phase 6 smoke test is green.
- Don't apply for HF Benchmarks allow-listing before Phase 12 — you'll be told to come back when you have ≥2 datasets and ≥2 models.
