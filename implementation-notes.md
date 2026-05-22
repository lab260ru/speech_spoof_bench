# Implementation Notes

Running notes captured during implementation of project specs/plans. Decisions, assumptions, tradeoffs, and deviations from the spec land here.

## 2026-05-21 — Phase 2 pip skeleton with local dataset loading

- Plan: `docs/plans/2026-05-21-phase-2-pip-skeleton.md`
- Spec: `docs/specs/2026-05-21-phase-2-pip-skeleton-design.md`
- Branch: `main` (user opted to stay on main per project convention; no PR/review flow yet)
- Execution mode: subagent-driven (fresh subagent per task + spec/quality review)

### Task 1 — package skeleton

- Changed: Code-quality reviewer flagged that the spec's `.gitignore` content dropped useful patterns from the pre-existing Phase 0 placeholder. Restored `env/`, `*.log`, `.mypy_cache/`, `.ruff_cache/`, `.vscode/`, `.idea/`, `*.swp` on top of the spec-listed entries; the spec is a minimum, not an exhaustive list.
- Validation: `pip install -e ".[dev]"` succeeded; `python -c "import speech_spoof_bench"` printed `0.1.0`.

### Task 2 — metrics + EER

- Changed: Plan's reference EER implementation was O(N²) — the FAR/FRR sweep used Python list comprehensions `[(spoof >= t).mean() for t in thresholds]`. Measured ~7.5s on LA's 71k thresholds (reviewer ran the actual benchmark). Vectorized using `np.sort` + `np.searchsorted` → ~0.04s on 100k items (700× speedup). Algorithm semantics preserved exactly: same threshold sweep, same crossover interpolation.
- Tradeoff: Skipped reviewer's minor suggestions (rename `id` param to `metric_id`, add `_clear_registry_for_testing` hook). YAGNI for Phase 2.
- Validation: 6/6 metric tests pass (4 original + 2 added for empty-bona/empty-spoof `ValueError`).

### Task 5 — runner

- Decision: Reviewer flagged that the plan's `test_per_item_skip_only_offender_in_multi_item_batch` used 8 rows + 1 bad item — but 1/8 = 12.5% trips the 5% TooManySkips threshold. Initial implementer hacked around it by adding `_MIN_SKIPS_FOR_THRESHOLD = 2` to the runner; that silently weakened production behavior. Reverted: grew test to 40 rows (1/40 = 2.5% < 5%) and restored the runner to the spec's two-condition check.
- Changed: After code-review nits — hoisted `from math import gcd` out of the inner branch; dropped dead `bad_index`/`_force_raise` parameter from test helper; added a length check on `model.score_batch` return value so a misbehaving model that returns the wrong number of scores triggers the per-item fallback instead of silently dropping items.
- Validation: 21/21 tests pass.

### Task 11 — end-to-end smoke test

- Changed: First smoke-test run failed because `cli.py` never imported the metrics module, so the registry was empty and the loader raised `KeyError: "metric id 'eer_percent' not registered"`. The plan assumed test files would import metric modules explicitly. Fixed by adding `from . import eer` at the bottom of `src/speech_spoof_bench/metrics/__init__.py` so any code path that imports the package gets the built-in metrics auto-registered. New metric files need a one-line addition here to participate.
- Validation: smoke test against `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/` produced `scores.txt` with 71237 lines (matches official LA eval count), `result.yaml` with `eer_percent: 49.87` (random baseline jitter near 50%), `dataset.id: ASVspoof2019_LA`, `dataset.revision: null`, sha256 64-char hex, `bench_version: speech-spoof-bench==0.1.0`. Resumability check: second run with default `skip_existing=True` completed in 4.3s (Python+datasets startup overhead; actual skip is instant) with `INFO skipping ASVspoof2019_LA (result.yaml present)`.
- Not verified: HF-id dispatch path (`validate-dataset SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA`) skipped to avoid pulling a 460 MB parquet shard. Local-mode is the primary user-facing path per the original ask; HF-mode wiring has unit-test coverage via the monkeypatched dispatch test in `tests/test_loader.py`.

## 2026-05-21 — Phase 4 arena-manifest

- Plan: `docs/plans/2026-05-21-phase-4-manifest.md`
- Spec: `docs/specs/2026-05-21-phase-4-manifest-design.md`
- Execution mode: subagent-driven (fresh subagent per task + spec/quality review)
- Scope spans two repos: pip package (this repo) + `arena-manifest/` (sibling repo at `/home/kirill/speech-spoof-bench/arena-manifest/`)

### Task 1 — publish manifest.yaml

- Validation: LA HEAD re-confirmed at `9b2040e8c57749dcd9a4f16ad61b4f47626b89ec`. `arena-manifest` commit `ab59e25` pushed to HF; round-tripped via `hf_hub_download` byte-identical.

### Task 2 — JSON Schema

- Decision: revision regex is `^[a-f0-9]{7,40}$` (covers short + full sha). Bundled together with Task 3 in commit `380615c` rather than as a separate commit, per plan.

### Task 3 — manifest.py + tests

- Validation: 15/15 tests pass; module-level `hf_hub_download` import makes `monkeypatch.setattr(mf, "hf_hub_download", ...)` work cleanly.

### Task 4 — CLI

- Validation: 5/5 CLI tests pass. `_cmd_manifest` writes the raw downloaded file verbatim (no parse round-trip) so output matches upstream byte-for-byte.
- Subparser variable named `man` to avoid shadowing the `mf`/`manifest` module names.

### Task 5 — wheel packaging

- Validation: built wheel via `python -m build`; `unzip -l` confirms `speech_spoof_bench/schema/manifest.schema.json` (1416 bytes) is included.

### Task 6 — DoD verification

- Validation: 46/46 tests pass. `speech-spoof-bench manifest` against real HF matches the local `arena-manifest/manifest.yaml`. `speech-spoof-bench list` prints exactly `[core] SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA`. ROADMAP Phase 4 checkboxes ticked in commit `ef45ee5`.

Phase 4 commits in `speech-spoof-bench` repo: `380615c`, `ae6c36c`, `2aef9a0`, `ef45ee5`. Plus `ab59e25` in the `arena-manifest` repo.

## 2026-05-22 — Phase 6 end-to-end smoke test

- Spec: `docs/specs/2026-05-22-phase6-smoke-test.md`
- Plan: `docs/plans/2026-05-22-phase6-smoke-test.md`
- Execution mode: hybrid — one subagent for the three autonomous checks (sha parity, rerun reproducibility, EER parity); inline for the three browser-verified checks (cold-start, edit/refresh, malformed YAML)
- Rationale: this phase writes no code, so the implementer+spec+quality review loop in subagent-driven-development would burn tokens reviewing nothing. Browser-verified tasks also need live human pauses that subagents cannot do.
- Spec amendment mid-flight: original spec said the random baseline was unseeded and "we do not re-run"; checking `examples/random_baseline.py` showed `seed=0` default, so the spec was updated to include a regeneration check (Task 2 in the plan).

### Results

- Check 1 (sha parity, 3-way): PASS. Local, model-repo (`hf_hub_download` at pinned commit `f63c30b`), and YAML `artifact.scores_sha256` all = `71ac000c…55587`.
- Check 1b (rerun reproducibility): PASS. `speech-spoof-bench run --output-dir /tmp/phase6-rerun --datasets <local LA dir>` produced bit-identical `scores.txt`. Confirmed CLI supports `--output-dir`; `--datasets` accepts local paths (no `--data-dir` flag needed — avoided HF download).
- Check 2 (EER parity): PASS. `compute_eer` returned `49.870836165873556` exactly. Note: `MetricResult.value` for `eer_percent` is already in percent units (not fraction). LA labels exposed as `path` + `label` (ClassLabel bonafide=0/spoof=1); subagent stripped `.flac` suffix to align utt_ids with scores.txt — 71237/71237 matched.
- Check 3 (Arena cold-start): PASS (user-confirmed in browser).
- Check 4 (edit + refresh round-trip): PASS. Bumped `eer_percent` +1.00 in `submissions/random-baseline.yaml` on LA dataset HF repo, user confirmed visible after Refresh, then `git revert HEAD` restored it. Two LA-repo commits added then reverted.
- Check 5 (malformed YAML resilience): PASS. Pushed `submissions/broken-test.yaml` with unterminated string. User confirmed via Arena Refresh: random-baseline row still visible AND About tab surfaced the `ScannerError` warning. Deleted the broken file with a follow-up commit.

### Validation summary

All five Phase 6 checks PASS. Ingest is resilient to per-file YAML errors (one bad file → warning, doesn't sink the rest). The seeded random baseline reproduces bit-exact, so the artifact chain from local → model-repo → submission YAML → Arena is verifiably consistent. No code changed. Phase 6 checkboxes ticked in `ROADMAP.md`. Ready for Phase 7.

### Follow-ups / not verified

- `cli.py`'s `validate-dataset` is still a Phase 2 stub; full schema/sha/EER drift checks come in Phase 7. The Phase 6 verification ran these checks ad-hoc; Phase 7 should formalize them in `reproduce --scoring`.
- LA dataset working copy at `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA` accumulated 4 smoke-test commits today (bump, revert, broken add, broken remove). They're all on `main`; left as-is for audit trail.

