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
