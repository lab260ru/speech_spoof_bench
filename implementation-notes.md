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
