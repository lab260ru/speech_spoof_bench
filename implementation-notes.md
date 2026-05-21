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
