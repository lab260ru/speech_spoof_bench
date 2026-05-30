# Testing & Pitfalls — "It Worked While I Was Developing"

This doc exists for one reason the project owner asked for directly: **stop changes that
pass locally but break in production.** It is split into a *verification matrix* (run
these before you call anything done) and a *catalogue of silent breakages* (the things
that pass tests, pass locally, and fail live).

The recurring theme: this system has **four independent artifacts** (package, manifest,
Arena, dataset/model repos) that are **loosely coupled by pins and commit SHAs**. A change
in one is not seen by the others until something is bumped or refreshed. Most failures
live in those gaps.

---

## The verification matrix

Match the change you made to its column; run everything in that column.

| Check | Model | Dataset | Metric | Package | Manifest | Arena |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|
| `pytest` (package) green | | | ✅ | ✅ | | |
| `pytest` (arena) green | | | | | | ✅ |
| Random-baseline smoke run ≈ 50% EER | ✅ | ✅ | ✅ | ✅ | | |
| `validate-dataset` all-green **offline** | | ✅ | ✅ | | | |
| `validate-dataset` green **online** (`org/name`) | | ✅ | | | ✅ | |
| `reproduce --scoring --no-local` matches | ✅ | ✅ | ✅ | ✅ | | |
| Ran once with `--no-local` (canonical revision) | ✅ | ✅ | | ✅ | | |
| Schema `const` + fixtures updated together | | | | ✅ | ✅ | |
| Version bumped (`pyproject` + `__init__`) | | | ✅ | ✅ | | |
| `ranking_version` bumped + CHANGELOG entry | | | | | ✅ | |
| Arena pin bumped in `requirements.txt` | | | ✅* | ✅* | | ✅ |
| Manifest still validates against schema | | ✅ | | ✅ | ✅ | |
| All 7 Arena tabs render on the live Space | | ✅ | ✅ | ✅ | ✅ | ✅ |
| Badge endpoint responds (`/badge/<slug>/tier.json`) | | | | | | ✅ |

\* only when the package change affects schema / validation / ranking-relevant logic the
Space needs.

---

## Catalogue of silent breakages

### Audio & scoring (model dev)

- **Score direction flipped.** Higher must mean *more bona fide*; label 0 = bona fide.
  Backwards gives EER ≈ `100 − true` (e.g. 97% not 3%). Looks like a "bad model", is
  actually a sign error. *Catch:* sanity-check EER magnitude; the random baseline gives ~50%.
- **Resampling twice.** You're already handed float32 mono at `expected_sample_rate`.
  Resampling again shifts numbers subtly — reproduce will then disagree with you. *Catch:*
  never resample in `score()`.
- **Heavy work in `__init__` instead of `load()`.** Breaks `--model-module` import and
  defeats the once-per-run load contract. *Catch:* construct cheaply; load in `load()`.
- **>5% of items skipped → `TooManySkips` aborts the dataset.** A `score()` that throws on
  some inputs silently raises the skip count until the whole run dies. *Catch:* watch
  `n_skipped`; handle edge cases in `score()`.
- **Batch-only bug hidden by the per-item fallback.** If `score_batch` raises, the runner
  silently retries one-at-a-time — correct results, mysterious slowdown. *Catch:* test with
  a real `batch_size > 1`.

### Datasets

- **Unstable `utterance_id`.** Re-sharding that changes ids breaks D5 and every
  submission's coverage check. *Catch:* derive ids deterministically; never depend on row order.
- **`notes` not JSON / missing `utterance_id`.** One bad row fails D4 and aborts. *Catch:*
  `json.dumps({"utterance_id": ...})` for every row; validate offline first.
- **Wrong `label` feature.** Must be `ClassLabel(["bonafide","spoof"])` in that order
  (bonafide=0). A plain string/int column trips D2. 
- **Sub-second clips / wrong sample rate.** D3 wants ≥1.0 s at 16 kHz.
- **Missing `arena-ready` tag or a D6 front-matter key.** The dataset validates as "not
  arena-ready" and the Arena won't surface it. *Catch:* D6 lists every required key.
- **Column projection myth.** In `reproduce`/ingest, labels are loaded with
  `load_dataset(..., columns=[...])` for *real* network pushdown. A post-load
  `select_columns()` still transfers the audio bytes — slow, not wrong, but it'll look like
  reproduce is "hanging" on big datasets. (Applies to `validate-dataset`'s D4/D5 scan too:
  dropping the audio via `select_columns` skips the *decode* — a big win when reading local
  parquet — but the online scan still transfers audio unless the columns are pushed down at
  `load_dataset` time.)

### Submitting (`submit`)

- **`submit` + a locally-registered dataset passed by `org/name` 404s.** When the dataset
  is in the local registry (`speech-spoof-bench local list`), `submit` resolves it to the
  local source whose `canonical_id` is the **bare directory basename** (e.g.
  `ASVspoof2021_LA`, org prefix dropped), then calls `repo_info("ASVspoof2021_LA")` →
  `RepositoryNotFoundError`. *Catch / workaround:* run `submit --no-local`, or use the
  manual upload path in [new-model.md](new-model.md#submitting-manually-when-submit-would-re-stream).
- **`submit --no-local` re-scores from scratch and re-streams the audio.** It does not
  reuse an existing `results/<DATASET>/scores.txt`, so it re-downloads the full dataset and
  re-runs the model (pure waste for a baseline that ignores audio). Worse, if you cancel it
  mid-run it can leave a **truncated** `scores.txt` (e.g. 68k of 181k lines) whose sha no
  longer matches `result.yaml`. *Catch:* prefer the manual upload path when you already have
  scores; if you do cancel a `submit`, regenerate `scores.txt` cleanly before reusing it.

### Local dev environment

- **`local-datasets.yaml` in the repo root breaks `pytest`.** `local_registry._REGISTRY_PATH`
  is `Path("local-datasets.yaml")` — **relative to the CWD**. Run `speech-spoof-bench local
  set …` from the package repo and you create that file in the repo root; a conftest guard
  then trips (`patched read_text used without writing first`) because a test resolved a
  dataset and read the real registry. CI is green because a clean checkout has no such file.
  *Catch:* the file is not gitignored — keep it out of the package repo root (run `local set`
  from elsewhere, or move the file aside before `pytest`), or gitignore it.

### Schemas & versioning

- **Editing a schema shape without bumping the `const`** (or vice-versa). Tests assert the
  `const`; mismatches between schema, producer (`submit.py`/`badge.py`), and fixtures slip
  through if you only touch one. *Catch:* the "changing a schema" checklist in
  [contributing-package.md](contributing-package.md).
- **Forgetting `invalid_wrong_schema_version.yaml`.** This fixture must stay one version
  behind, proving old versions are rejected. Bump the schema, forget the fixture, and a
  real regression can pass.
- **`manifest.schema.json` is consumed by two repos.** Bump it and the Arena's manifest
  reader breaks unless the Arena is updated and its pin bumped *in lockstep*.
- **Capitalised dataset SHA.** `revision` must match `^[0-9a-f]{7,40}$` — lowercase hex.
  A SHA copied from a capitalising tool fails manifest validation.
- **`ranking_version` typo / trailing space.** It's a free string with no pattern; `"v2 "`
  is "valid" but wrong. Keep it `v1`, `v2`, ...

### The pin & refresh gaps (the big ones)

- **Arena runs stale package code.** The Space pins `speech-spoof-bench @ …@<sha>`. A
  package change you released is *invisible* to the live Arena until that pin is bumped.
  Symptom: schema/validation/ranking behaves like the old version on the live site. This is
  the project's #1 recorded gotcha (`arena_package_pin`). *Catch:* bump
  `arena/requirements.txt` whenever package schema/logic changes.
- **Submit tab shows stale docs.** `docs_fetch` pins the submit guides to the same package
  SHA from `requirements.txt`. Update the guides but not the pin → the Space still shows the
  old guides.
- **New manifest dataset not webhook-routed yet.** The webhook's subscription set is the
  in-memory manifest; a newly-added dataset isn't routed until an ingest `force_refresh`
  runs. Brief "not subscribed" window after adding a dataset.
- **Freshly merged submission not on the board for up to a refresh cycle.** Ingest TTL is
  30 min (60 s on failure) + the post-merge refresh. It's not instant — don't conclude the
  merge "didn't work".

### CI / secrets

- **`HF_BOT_TOKEN` missing → verdicts/badges print to stdout, not HF.** No error, nothing
  posted. The Action goes green and the discussion stays empty. *Catch:* confirm the
  comment actually appears on the PR.
- **`GH_PAT` / `HF_WEBHOOK_SECRET` / `SPACE_COMMIT_TOKEN` missing.** Respectively: dispatch
  silently skipped, every webhook 401s, cache never persists across restarts. All soft
  failures (logged, not crashed).
- **Webhook ↔ workflow drift.** Rename a workflow file or change its `inputs`, and the
  `webhook.py` dispatch URL/inputs must change too — otherwise dispatch 404s silently.
- **Badge sentinel mismatch.** The `<!-- ssb:badge --> sha=… path=…` string must be
  byte-identical between the writer and the duplicate-check, or every merge posts a
  duplicate badge.
- **`reproduce` tolerance is hardcoded `1e-6` in two places** (`verify_pr.py`,
  `nightly.py`). Change the contract in one and not the other and verification disagrees
  with nightly.
- **No retries.** A transient HF/network blip fails that one PR verification. Re-dispatch
  the workflow.
- **HF webhook v3 PR-number recovery is regex-fragile.** Merge events recover the PR number
  from the commit-title `(#N)` suffix. If HF changes that format, post-merge badges quietly
  stop. (See [../architecture/cicd.md](../architecture/cicd.md).)

### Arena internals

- **Sticky-column widths hardcoded** (`_PINNED_WIDTHS = [44, 150]`). Widen Rank/System and
  the frozen columns misalign.
- **Paper-gating means rows can have no rank.** When any tier `requires_paper`, paperless
  systems are dropped from the ranking dict and their tier shows no Rank column. Don't
  assume every `Row` has a `place`.
- **Cache hash must stay stable.** `cache_store` hashes sorted rows/warnings excluding
  `loaded_at` so timestamp/order noise doesn't trigger commits. Change the sort keys and the
  debounce breaks (commit storms or missed commits).
- **Ingest holds a lock for the whole refresh.** A slow HF fetch blocks badge/feed requests.
  Don't add long synchronous work inside the locked section.

---

## The minimum bar before "done"

For a **model submission:** offline run sane → `reproduce --scoring --no-local` matches →
PR opened → the ✅/❌ verify comment is green on the discussion.

For a **dataset:** `validate-dataset` all-green offline → pushed → green online → manifest
PR with a pinned lowercase SHA.

For a **package change:** `pytest` green → version bumped in both files → schema/fixtures
in sync → Arena pin bumped if the Space needs it.

For a **manifest change:** validates against the schema → `ranking_version` bumped if rules
changed → CHANGELOG entry → all 7 Arena tabs verified on the live Space.

If you can't tick the relevant row of the matrix, it isn't done — it just *looks* done on
your machine.
</content>
