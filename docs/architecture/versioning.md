# Versioning, Revisions & When to Update

This is the single most important operational doc. The system has **several independent
version tracks**, each governing a different thing and bumped by a different trigger.
Mixing them up is the most common way to ship a silently-broken change. Read the trigger
matrix at the bottom before you touch any version number.

## The version tracks

| # | Version | Where it lives | Type | Current | Governs |
|---|---------|----------------|------|---------|---------|
| 1 | **Package version** | `pyproject.toml` + `__init__.py` `__version__` | SemVer | `0.1.0` | The pip package's code (eval engine, CLI, schemas-as-shipped). |
| 2 | **Manifest schema version** | `manifest.yaml` `schema_version`, enforced by `manifest.schema.json` | int `const` | `1` | The *shape* of `manifest.yaml`. |
| 3 | **Ranking version** | `manifest.yaml` `ranking_version` | opaque string | `v2` | Tier/ranking *rules and values*. |
| 4 | **Submission schema version** | submission YAML `schema_version`, enforced by `submission.schema.json` | int `const` | `4` | The shape of a submission. |
| 5 | **Result schema version** | result YAML `schema_version`, enforced by `result.schema.json` | int `const` | `1` | The shape of the badge-layer result. |
| 6 | **Dataset revision** | `manifest.yaml` `core_set[].revision` / `extended[].revision` | git SHA (7–40 hex) | per-dataset | Which snapshot of a dataset is "official". |
| 7 | **Cache schema version** | `arena/cache.json` `schema_version` | int | `1` | The Arena cache snapshot shape. |
| 8 | **Local registry schema** | `local-datasets.yaml` `schema_version` | int | `1` | The offline registry file shape. |
| 9 | **Arena package pin** | `arena/requirements.txt` | git SHA | `bde5949…` | Which package commit the live Space runs. |
| 10 | **bench_version** | written into every result/submission `artifact.bench_version` | derived from #1 | `speech-spoof-bench==0.1.0` | Provenance: which package produced a score. |

> Note `arena/pyproject.toml` has `version = "0.0.0"` — a placeholder. The Arena is **not**
> SemVer-versioned; it ships from `main`. Its only real "version" is the package pin (#9).

## How they relate

```
Package version (SemVer) ──┐
   bumps independently      ├─► bench_version stamped into every result.yaml
                            │
manifest.schema.json (1) ──┤  locked; bump only on shape change → needs pip + manifest update
                            │
ranking_version (v2) ──────┤  bumps on rule/value change → manifest-only, triggers re-rank + badge refresh
                            │
submission schema (4) ─────┤  can bump to 5; old 4s stay valid unless deprecated
                            │
dataset revision ──────────┘  per-dataset; old revisions stay reproducible forever
```

**Loose coupling is deliberate.** A ranking tweak is a manifest commit, no package
release. A new metric is a package release, no manifest change required. The Arena pin
(#9) is the one *manual* coupling: the Space doesn't auto-upgrade the package, giving the
maintainer a controlled cut-over point.

---

## Track-by-track: what bumps it, and when

### 1 + 10. Package version & `bench_version` (SemVer)
- **Bump when** package *code* changes: new metric, bug fix, EER algorithm change, CLI/API
  change. Patch (`0.1.1`) for fixes, minor (`0.2.0`) for additive features, major (`1.0.0`)
  for breaking changes.
- **Edit:** `pyproject.toml` `version` **and** `src/speech_spoof_bench/__init__.py`
  `__version__` (keep them identical — `bench_version` is derived from `__init__`).
- **Effect:** every new `result.yaml`/submission records the new `bench_version`. Old
  artifacts keep their old one. `reproduce --scoring` recomputes with *current* code; if a
  scoring change moved the numbers, reproduction of an old submission will (correctly)
  fail — that's the drift detector working.

### 2. Manifest schema version (locked at `1`)
- **Bump when** the *structure* of `manifest.yaml` changes — add/remove/rename a field,
  change a type. **Not** for data changes (new tier value, new dataset, new colour).
- **Edit (all together):** `schema/manifest.schema.json` `const`, the manifest, every
  consumer that reads the changed field (`manifest.py`, `arena/ingest.py`,
  `arena/ranking.py`), and `local_registry.py` `_SCHEMA_VERSION` if it mirrors. Then a
  **package release** (the schema ships in the wheel) **and** a manifest commit.
- Because it's a JSON-Schema `const`, any manifest with `schema_version != 1` is rejected
  at fetch time. There is no migration code — a bump is a breaking, coordinated change.

### 3. Ranking version (`ranking_version`, currently `v2`)
- **Bump when** ranking *behaviour* changes: tier thresholds (`min_coverage`), tier
  colours, `requires_paper`, adding/removing a tier, `gamma_aggregated`/`gamma_pooled`,
  `absence_penalty`, per-dataset `weights`, `default_view`, or the ranking `metric`.
- **Don't bump for:** adding a dataset to `core_set` (data), package changes.
- **Edit:** `manifest.yaml` (`ranking_version` + the changed rule) and add a
  `CHANGELOG.yaml` entry. No package release needed. Commit (and tag) the
  `arena-manifest` repo.
- **Effect:** the Arena re-fetches and re-ranks everything under the new rules; tiers and
  ranks may move. Old submissions stay valid (their `schema_version` is untouched).
- Format is a free string — keep it simple (`v1`, `v2`, `v3`). It has no schema pattern,
  so a typo like `"v2 "` won't be caught; be careful.

### 4. Submission schema version (`4`)
- **Bump when** the submission shape changes (new required field, type/pattern change).
- **Edit:** `schema/submission.schema.json` `const`, `submit.build_submission_payload()`
  to emit the new version, the test fixtures (esp.
  `tests/fixtures/submissions/invalid_wrong_schema_version.yaml`), then a package release.
- Old submissions at v4 stay valid unless explicitly deprecated; the design allows v5.

### 5. Result schema version (`1`)
- **Bump when** the badge-layer result shape changes. Independent of #4.
- **Edit:** `schema/result.schema.json` `const` and
  `badge._project_submission_for_result()`. Remember model cards already contain copies of
  the old shape — coordinate.

### 6. Dataset revision (git SHA, per dataset)
- **Bump when** a dataset's content changes: re-sharded parquet, added/edited `eval.yaml`
  metrics, fixed labels.
- **Edit:** commit the dataset repo, then update `manifest.yaml`
  `core_set[]/extended[].revision` to the new SHA, and add a `CHANGELOG.yaml` ↻
  `dataset_repin` note. (Re-pinning Core can change everyone's scores → consider whether
  it also warrants a `ranking_version` note for transparency.)
- Pattern: `^[0-9a-f]{7,40}$` — **lowercase hex only**. A capitalised SHA (Windows `git
  log`) fails validation.
- Old revisions remain fetchable forever, so old submissions stay reproducible.

### 7. Cache schema version (`1`)
- **Bump when** `cache.json`'s structure changes. There is currently **no migration
  logic** — `cache_store._deserialize()` would need one. In practice a stale cache is
  harmless: a refresh rebuilds it.

### 8. Local registry schema (`1`)
- Internal/offline only. Bump only if it ever needs to mirror a manifest-schema change.

### 9. Arena package pin (`arena/requirements.txt`)
- **Bump when** the Space needs new package behaviour — and **always** after a package
  change that affects schema/validation/ranking-relevant logic, or the Space runs stale
  code against current data.
- **Edit:** the `speech-spoof-bench @ git+https://github.com/lab260ru/speech_spoof_bench.git@<sha>`
  line to the new commit SHA, then push `arena/`. This same SHA is what `docs_fetch`
  uses to show the matching submit guides, so bumping it also updates the Submit tab.
- This is the project's recorded gotcha (`arena_package_pin` in memory): *"bump the
  speech-spoof-bench @sha in arena/requirements.txt when package schema/logic changes,
  else the Space runs stale code."*

---

## The trigger matrix

| Change you're making | Bump | File(s) to edit | Release/Tag |
|----------------------|------|-----------------|-------------|
| New metric (e.g. min-tDCF) | Package minor `0.1.0→0.2.0` | `metrics/<new>.py`, `metrics/__init__.py` import, `__init__.py`, `pyproject.toml`, test | pip tag → then bump Arena pin (#9) |
| Bug fix in scoring | Package patch `0.1.0→0.1.1` | the module + `__init__.py` + `pyproject.toml` | pip tag → bump Arena pin |
| EER algorithm change | Package minor/major | `metrics/eer.py` + version | pip tag → bump Arena pin |
| Tier threshold / colour / `requires_paper` | `ranking_version` `v2→v3` | `manifest.yaml`, `CHANGELOG.yaml` | arena-manifest tag |
| Gamma / absence_penalty / weights | `ranking_version` | `manifest.yaml` | arena-manifest tag |
| Add a tier | `ranking_version` (+ schema if new field) | `manifest.yaml` (+ schema) | arena-manifest tag (+ pip if schema) |
| Add field to manifest | `schema_version` `1→2` | `manifest.schema.json`, `manifest.yaml`, all consumers | pip tag **and** arena-manifest tag |
| Add field to submission | submission schema `4→5` | `submission.schema.json`, `submit.py`, fixtures | pip tag |
| Add dataset to Core/Extended | **no version bump** | `manifest.yaml` `core_set`/`extended` | arena-manifest commit |
| Re-shard / re-pin a dataset | dataset revision (SHA) | `manifest.yaml` revision, `CHANGELOG.yaml` | dataset repo commit |
| Arena UI change | **no version bump** | `arena/*.py` | push `arena/` (untagged) |
| New package commit the Space should run | Arena pin (#9) | `arena/requirements.txt` | push `arena/` |

## Who owns what

- **Package maintainer** — package version, schemas, `submit.py`/`benchmark.py`; tags the
  pip repo.
- **Manifest maintainer** — `manifest.yaml`, `CHANGELOG.yaml`; tags `arena-manifest`.
- **Dataset maintainer** — dataset repo contents; updates the manifest revision when the
  official snapshot changes.
- **Arena dev** — `arena/*` (untagged) and the package pin.

## Reproducibility invariant

A leaderboard number is reproducible because **all four of its inputs are pinned**: the
dataset (`dataset.revision` SHA), the score file (`scores_url` commit + `scores_sha256`),
the metric code (`bench_version`), and the ranking rules (`ranking_version`). Change any
of them and you've changed the number — which is exactly why each has its own version
track and its own "when to bump" rule.
</content>
