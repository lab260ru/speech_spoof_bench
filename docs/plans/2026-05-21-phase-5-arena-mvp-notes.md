# Phase 5 — Arena MVP Implementation Notes

Running notes for the Phase 5 Arena MVP implementation.

Plan: `docs/plans/2026-05-21-phase-5-arena-mvp.md`
Spec: `docs/specs/2026-05-21-phase-5-arena-mvp-design.md`

---

## 2026-05-21 — Kickoff

- Execution mode: subagent-driven (fresh subagent per task, review between).
- Working dirs: `/home/kirill/speech-spoof-bench/arena/` (HF Space sources) and `/home/kirill/speech-spoof-bench/speech-spoof-bench/` (pip package; Tasks 1–2 add `submission.schema.json` + `submission.py`).
- Assumption: pip package GitHub URL in `arena/requirements.txt` is `https://github.com/lab260ru/speech_spoof_bench.git`. To be confirmed at Task 3 or Task 10 (pin step). If wrong, fix in place.
- Assumption: HF Space repo lives at `huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena` (user-confirmed). Already initialized with README/SDK frontmatter.

## 2026-05-21 — Task 1: submission.schema.json

- Validation: schema validates the real `random-baseline.yaml` once dates are coerced to strings (PyYAML returns `datetime.date` for bare YAML dates).
- Follow-up: `submission.py` (Task 2) must coerce `datetime.date` → `str` before calling `jsonschema.validate`. The schema is correct as written (JSON has no date type).
- Commit: `e70ee8b` in `speech-spoof-bench/`.

## 2026-05-21 — Task 2: submission.py

- Decision: added `_coerce_dates` helper that recursively turns `datetime.date` → ISO strings before validation. Required because PyYAML produces `datetime.date` for bare YAML dates and jsonschema's `format: date` validator expects strings.
- Validation: `pytest tests/test_submission.py` → 4/4 passed.
- Commit: `ef50b5a` in `speech-spoof-bench/`.

## 2026-05-21 — Task 3: bootstrap arena

- Tradeoff: kept `requires-python = ">=3.11"` in `arena/pyproject.toml` (matches HF Space `python_version: 3.13`). Local dev machine has Python 3.10; installs require `--ignore-requires-python`. Code uses `from __future__ import annotations` so runs fine on 3.10 in practice. Lowering the constraint is a possible follow-up if local dev friction becomes a problem.
- Decision: dropped the `missing-reproduction.yaml` fixture — `schema-invalid.yaml` (no `reproduction:` block) IS the missing-reproduction case (schema requires it).
- Validation: `pytest --collect-only` → "no tests collected", clean.
- Commit: `9bd509e` in `arena/`.

## 2026-05-21 — Tasks 4–8: arena code

- Tasks 4-7 commits in `arena/`: `dae1d6f` (schema dataclasses), `143739f` (assign_tiers), `8476c26` (table builders), `596c795` (ingest with TTL cache).
- Task 8 commit: `ac68507` (Gradio app.py). Subagent did import-only smoke check; live launch deferred to Task 9 (UI dialog with user).
- All ranking/ingest tests green (10 + 5).

## 2026-05-21 — Task 9: interactive UI review

- Two fixes surfaced during the walkthrough:
  1. `gr.DataFrame` was rendering list-of-dicts as a single "[object Object]" column. Wrapped output in `pandas.DataFrame` via `_to_df` helper (so `pandas` joins the runtime dep set — already pulled in by gradio).
  2. About tab markdown collapsed single newlines into one line. Restructured `_about_text` to emit a bullet list + blank-line-separated paragraphs.
- Commit: `30448cf` in `arena/`.
- Validation: live UI checked end-to-end on http://127.0.0.1:7861 with the user. All five Overview steps, all three Per dataset steps, both About checks, and Refresh button — all green.
- Follow-up: `pandas` is not listed in `arena/requirements.txt` explicitly. It's transitively pulled in by `gradio`, so the HF Space build will get it for free. Consider adding it explicitly if any future requirements bump drops it.

## 2026-05-21 — Task 10: deploy to HF Space

- Pushed pip pkg commits `e70ee8b`, `ef50b5a` to `lab260ru/speech_spoof_bench` (GitHub `origin/main`).
- Pinned `arena/requirements.txt` to sha `ef50b5a3ec55e6f80f1a5f1440939e85b60f5102`. Commit `2cfd44c`.
- Pushed arena to `huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena` (`origin/main`).
- First load failed with `ValueError: not enough values to unpack (expected 3, got 0)` from `_overview_tables` — because the manifest fetch returned empty when the dataset repos were still private.
- Resolution: user flipped `arena-manifest` and `ASVspoof2019_LA` dataset repos to public; Space worked on next load. No code change needed. (Tried a defensive empty-manifest fallback first, then reverted since the actual fix is the repo visibility.)
- Follow-up: dataset / manifest repos must stay public for the Space to function (documented in spec §5 — the arena does anonymous HF reads).
