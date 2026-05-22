# Phase 6 — End-to-end smoke test

**Date**: 2026-05-22
**Roadmap reference**: `docs/roadmap/ROADMAP.md` § Phase 6
**Goal**: verify the Phase 1–5 data path works end-to-end before adding CI/webhook automation.

## Context

Phases 1–5 produced:
- LA dataset on HF in v4 form (`SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA`).
- Pip package with `run` working locally.
- One hand-authored submission YAML (`submissions/random-baseline.yaml`) pointing to scores in `SpeechAntiSpoofingBenchmarks/random-baseline-asas`.
- Manifest at `SpeechAntiSpoofingBenchmarks/arena-manifest`.
- Arena Gradio Space at `SpeechAntiSpoofingBenchmarks/arena` (read-only, manual refresh).

No automation yet. This phase only verifies the chain holds together.

## Scope

Five checks. Three are autonomous (Claude executes). Two require browser verification by the user.

### Check 1 — scores.txt parity (autonomous)

- Download `scores.txt` from the model repo (`SpeechAntiSpoofingBenchmarks/random-baseline-asas` at path `.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt`) via `huggingface_hub.hf_hub_download`.
- `sha256sum` the downloaded file.
- Read `scores_sha256` from `submissions/random-baseline.yaml` in the LA dataset repo.
- Assert all three match: downloaded-file sha == YAML's `scores_sha256` == local `results/ASVspoof2019_LA/scores.txt` sha.

Note: the random baseline is unseeded, so re-running `speech-spoof-bench run` would produce different scores. We do not re-run here; we verify the artifact chain (local → model repo → submission YAML sha) is consistent.

### Check 2 — EER parity (autonomous)

- Load the local `scores.txt` from `results/ASVspoof2019_LA/scores.txt`.
- Load LA labels via `datasets.load_dataset(..., split="test")`.
- Compute EER using the package's `eer_percent` metric.
- Read `scores.eer_percent` from `submissions/random-baseline.yaml`.
- Assert exact match (both derive from the same scores file).

### Check 3 — Arena cold-start (manual)

- User opens the Space URL in browser.
- User confirms: random-baseline row visible on Overview tab; per-dataset tab shows LA with the published EER; no error banners.

### Check 4 — Edit + Refresh round-trip (manual + autonomous)

- Claude: push a single-line change to `submissions/random-baseline.yaml` on the LA dataset HF repo bumping `scores.eer_percent` by `+1.00`. Use a descriptive commit message so the revert is obvious.
- User: hit "Refresh" button in Space, confirm new EER appears.
- Claude: revert the commit (push a second commit restoring original value) immediately after user confirms.
- User: hit "Refresh" again, confirm original value returns.

### Check 5 — Malformed YAML resilience (manual + autonomous)

- Claude: push a deliberately-malformed `submissions/broken-test.yaml` (invalid YAML syntax — e.g. unclosed bracket) to the LA dataset HF repo.
- User: hit "Refresh", confirm:
  - random-baseline row still appears (one bad file doesn't break ingest);
  - About tab surfaces a warning naming the broken file (per Phase 5 ingest spec).
- Claude: delete `submissions/broken-test.yaml` from the LA dataset HF repo immediately after user confirms.

## Exit criteria

All five checks pass. Failures block Phase 7 — fix root cause before proceeding.

## Out of scope

- Seeding the random baseline for reproducible regeneration (would belong in Phase 7's `reproduce --scoring`).
- Automated CI of any kind (Phase 8).
- Adding a second dataset or model (Phases 11–12).
