# Contributing to the Package

Working on `speech_spoof_bench` itself — fixing a bug, changing the eval engine, evolving
a schema, cutting a release. The package is the brain; a careless change here can silently
break the live Arena (which pins a specific commit) and every CI workflow (which
`pip install -e .`s `main`).

## Ground rules

1. **Tests are the contract.** `pytest` must be green before *and* after. The suite
   encodes the schema versions, the EER reference value, the validation checks, and the CI
   logic. If a test changes, that change *is* the spec change — make it deliberately.
2. **Schemas are versioned contracts**, not config. Editing a `schema/*.json` shape is a
   version bump with downstream coordination (see below and
   [../architecture/versioning.md](../architecture/versioning.md)).
3. **The Arena pins a commit.** Your change reaches production only when someone bumps
   `arena/requirements.txt`. Until then the Space runs the old code against current data.
   Plan the cut-over.

## Project layout for contributors

```
src/speech_spoof_bench/
├── model.py            # the model ABC — changing it changes the contributor contract
├── benchmark.py        # Benchmark.run orchestration + result.yaml writer
├── runner.py           # per-dataset loop, resampling, batching, scores.txt
├── loader.py           # dataset resolution (local/registry/HF)
├── metrics/            # the metric registry (+ eer.py)
├── submission.py       # parse/validate/fetch submissions
├── validate.py         # D1–D7 / S1–S4 checks
├── submit.py           # run + upload + open PR
├── scaffold.py         # dataset skeleton generator
├── reproduce.py        # run_scoring (the verification gate)
├── badge.py            # result.yaml projection + paste comment
├── local_registry.py   # offline dataset map
├── manifest.py / hf_fetch.py / cache.py
├── ci/                 # verify_pr, nightly, post_merge_badge (called by GH Actions)
├── schema/*.json       # manifest / submission / result schemas (bundled)
└── data/               # submission_meta schema + dataset_skeleton template (bundled)
tests/                  # mirrors the above; tests/ci uses mocks
```

## The test suite, by area

```bash
pytest                              # everything
pytest tests/metrics/               # EER correctness (test_eer.py uses scores_known.txt)
pytest tests/test_validate_dataset.py tests/test_validate_submission.py
pytest tests/test_submission_schema.py tests/test_badge_result_schema.py   # schema consts
pytest tests/ci/                    # verify-pr / nightly / post-merge-badge (HfApi & gh mocked)
pytest tests/test_reproduce*.py     # the reproduction gate
```

`tests/ci/` mocks `HfApi`, the `gh` CLI, and `subprocess` — so it never touches the
network. Keep the monkeypatch seams in place (e.g. `ingest`-style indirection functions);
inlining them breaks the tests.

## Changing a schema (the careful part)

Each schema pins its `schema_version` as a JSON-Schema `const`, and at least one test
asserts that const. To change a shape:

1. Edit `schema/<x>.schema.json`: the shape **and** bump the `const`.
2. Update the producer:
   - `submission.schema.json` → `submit.build_submission_payload()` (emits the version).
   - `result.schema.json` → `badge._project_submission_for_result()`.
   - `manifest.schema.json` → every reader (`manifest.py`, and over in the Arena
     `ingest.py`/`ranking.py`); also `local_registry._SCHEMA_VERSION` if it mirrors.
3. Update the **fixtures**: e.g. `tests/fixtures/submissions/invalid_wrong_schema_version.yaml`
   exists to prove old versions are rejected. Keep it one behind the new const.
4. Bump the package version and tag.
5. Plan the migration for already-published artifacts (old submissions on HF, old
   `result.yaml`s in model cards). There is no automatic migration.

`manifest.schema.json` is the most expensive to change because both the package *and* the
Arena consume it — both must update together, and the manifest itself must be re-validated.

## Changing the eval engine

Touching `runner.py`, `benchmark.py`, or `metrics/eer.py` changes the *numbers*. That:

- Bumps the package version (patch for a fix, minor/major if results move).
- Is recorded in every new artifact's `bench_version`.
- **Will (correctly) make old submissions fail `reproduce --scoring`** if the change moved
  their numbers — that's the drift detector. Decide whether you intend that; if you do,
  re-verify the affected submissions and let nightly open `stale-submission` issues for the rest.

Add or update a reference case in `tests/metrics/` (the EER test pins a hand-checked value
from `tests/fixtures/scores_known.txt`).

## Cutting a release

1. Green `pytest`.
2. Bump `version` in `pyproject.toml` **and** `__version__` in `__init__.py` (identical).
3. Commit + tag in `lab260ru/speech_spoof_bench`.
4. If the change affects schema / validation / ranking-relevant logic, **bump
   `arena/requirements.txt`** to the new commit SHA and push `arena/` (this also updates
   the Submit tab, which `docs_fetch` pins to that SHA).
5. Update `CHANGELOG`/roadmap as appropriate.

## Verification before you call it done

- `pytest` green.
- `speech-spoof-bench --help` and the touched subcommand run.
- A smoke run with the random baseline still produces ~50% EER
  ([setup.md](setup.md)).
- If you touched CI logic: `pytest tests/ci/` green and re-read [../architecture/cicd.md](../architecture/cicd.md)
  for the coordinated edits (dispatch URLs, sentinel string, flag names).
- See the full catalogue in [testing-and-pitfalls.md](testing-and-pitfalls.md).
</content>
