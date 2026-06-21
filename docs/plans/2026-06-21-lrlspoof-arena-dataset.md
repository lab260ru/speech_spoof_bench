# Arena dataset plan: LRLspoof (spoof-only, labels-only, in-place) + the `srr_complement` metric

> This plan is broader than a normal dataset-add: LRLspoof is the **first spoof-only dataset**, so it
> also requires building the deferred `srr_complement` metric (package) and the per-dataset-metric
> rendering (arena) before the dataset can land. Three phases. The **one approval gate** is this plan.
>
> Source specs: `v2_notes/proposal.md` §5/§6, `v2_notes/stages/stage-4-schema-release.md`,
> `v2_notes/stages/stage-5-arena-code.md` §5, and the fully-worked code sketches in
> `v2_notes/plan.md` §C (srr_complement), §D (`--labels-only`), RANK §F (per-dataset metric).

---

# Plan (reviewed at the 🚦 PLAN GATE)

- **Dataset Name / manifest id:** `lab260/LRLspoof`  (kept in its existing org, **in-place** — user choice)
- **Raw source (read-only):** local extracted tree `/home/kirill/mnt/drive2_8tb/multilingual_mtuci_speech/lrl_spoof`; HF repo `lab260/LRLspoof` (dataset, public, sha `24eaac6b…`) where the audio lives as a **43-part split tarball** `lrl_spoof.tar.gz.part_aa…part_bq` (~500 GB, **never touched**)
- **Date:** 2026-06-21
- **Type:** spoof-only, labels-only, **in-place** (we only *add* tiny files to `lab260/LRLspoof`)

## What the dataset is
66 languages (low-resource focus) × multiple TTS systems. Layout: `<language>/<tts_model>/<file>.wav`,
with a sibling `<language>/<tts_model>.txt|.csv` (transcripts/metadata). Naming varies per model
(`line_59.wav` vs `1.wav`). **All utterances are spoof** (TTS-generated) → `label = 1` for every row.
`size_categories: 1M<n<10M` in the repo card → `n_trials` is ~1M+ (computed at build).

## Protocol → schema mapping (labels-only)
- **`utterance_id` = repo-relative POSIX path incl. extension**, e.g. `russian/xtts/1.wav`,
  `english/fastpitch/line_59.wav`. This is the join key between our `labels.parquet` and a submitter's
  `scores.txt`. **Assumption (to verify):** the tarball's internal paths equal the local tree's
  relative paths; the convention is documented prominently in `eval.yaml`/README so submitters match it.
- **label:** constant `1` (spoof) — no bonafide class.
- `data/labels.parquet` = `{utterance_id: string, label: int8=1}`, one row per `.wav`.

## Metric: spoof-only → `srr_complement` (1 − Spoof Rejection Rate)
- Lower-is-better, 0–100, spoof-only-safe. `SRR = mean(spoof_score < t*)`, value `= (1−SRR)·100`.
- `t*` is **calibrated on DeepVoice** per submission (the model's EER operating-point threshold on
  DeepVoice), carried in the submission's `calibration: {source_dataset, threshold}` block. Requires
  the model to have a DeepVoice submission. `random-baseline` already has one →
  `benchmarks/DeepVoice/submissions/random-baseline.yaml` (good — seed works).

## License & redistribution
- `license: mit` (matches the existing `lab260/LRLspoof` card; user owns the org). Redistribution:
  yes (user's own org / declared MIT). **Confirm at gate.**
- D6 requires an `arxiv` front-matter key. LRLspoof likely has **no paper** → need a value for that key
  (HF dataset DOI, or the repo URL). **Open item — confirm at gate** (see Open decisions #3).

## Manifest placement — **recommend `extended:` (not `core_set`)**
- **Why Extended:** adding a 20th *Core* dataset that only the baseline covers would (a) drop every
  full-coverage system below 1.0 coverage → **tier churn for ~11 systems** under the live 3-tier
  manifest, and (b) introduce a mixed-metric Core aggregate before the penalty/Platinum model
  (stage 5/6) is even deployed. Extended gives LRLspoof its **own per-dataset tab ranked on
  `srr_complement`** with **zero disruption** to the existing board, and it can be promoted to Core
  later once models have submitted and stage-5/6 ships. (proposal §5 says "keep spoof-only in Core" —
  but that's *within* the full penalty redesign, which is out of scope here.)
- Entry:
  ```yaml
  extended:
    - id: lab260/LRLspoof
      revision: <sha>
      n_trials: <count>
      metric: srr_complement
      category: spoof_only
      penalty: 50.0        # range-appropriate; tunable
  ```
  plus add `srr_complement` to top-level `metrics_in_use`. CHANGELOG `dataset_added` note.

## Build approach
- **No audio repackaging.** We enumerate the local tree (via a `benchmarks/LRLspoof` symlink the user
  OK'd) to emit `data/labels.parquet`, and **upload only** `data/labels.parquet` + `eval.yaml` +
  README front-matter edit + `submissions/` into `lab260/LRLspoof`. The 500 GB tarball is untouched.
- Enumeration uses `os.scandir` (sorted), not a recursive `find` (the earlier `find` was slow on the
  spinning disk and was interrupted); run in background if needed. Cross-check count vs. the sum of
  `<model>.txt`/`.csv` line counts.

## 🚦 PLAN GATE — approve the above (esp. the Open decisions). Nothing built/pushed before OK.

### Open decisions for you (recommendations in **bold**)
1. **Placement:** **Extended** (own tab, zero board disruption) vs Core (tier churn now). → recommend Extended.
2. **utterance_id convention:** **`<lang>/<model>/<file>.wav`** (relative POSIX path). OK, or prefer
   stem-only / a different normalization? (Must match what submitters derive from the extracted tarball.)
3. **D6 `arxiv` key (no paper?):** use the **HF dataset DOI** (I can mint one) or the repo URL, or do
   you have a paper/arXiv id? Needed for the dataset card to validate.
4. **penalty value:** **50.0** (default). Adjust later if the SRR range warrants.
5. **Scope confirm:** OK to build the package metric (`srr_complement` + `--labels-only`, one release)
   and the arena per-dataset-metric rendering as part of this — not just the dataset?

---

# Phase 1 — Package: `srr_complement` + `--labels-only` (one release)

Schemas are **already** at the needed versions (submission `enum:[4,5]` w/ `calibration` block at
`submission.schema.json:98`; manifest `const:2` w/ `metric`/`category`/`penalty` at
`manifest.schema.json:71-74`). So **no schema bump** — only metric logic + validate mode.

### 1A. `srr_complement` metric + `call_metric` contract  (plan.md §C)
- `metrics/__init__.py`: add `MetricConfig` alias; widen `MetricFn` to `Callable[..., MetricResult]`;
  add `call_metric(spec, scores, labels, config=None)` that passes `config` only if the fn takes a 3rd
  param (introspection guard) → **`eer.py` stays a 2-arg fn, untouched**. Add
  `from . import srr_complement` at the auto-import line (the one everyone forgets).
- New `metrics/srr_complement.py` exactly per plan.md §C.2 (raises if `config['threshold']` missing;
  uses only spoof rows → single-class safe).
- Route both callsites through `call_metric`: `benchmark.py:147`, `reproduce.py:236`.
- `reproduce.py`: add `_repro_config(data, mid)` reading `calibration.threshold`; **wrap the metric
  call in try/except** so a spoof-only submission *without* calibration **FAILs cleanly** (returns 1,
  no traceback) instead of crashing (plan.md §C.4/§C.5).
- Tests (plan.md §C.5): `tests/metrics/test_srr_complement.py`, `tests/metrics/test_call_metric.py`,
  reproduce srr test. Key props: random spoof-only at t=0 ≈ 50.0; eer back-compat unchanged.

### 1B. `validate-dataset --labels-only`  (plan.md §D)
- `cli.py`: add `--labels-only` + `--n-trials` flags; thread into `validate.validate_dataset`.
- `validate.py`: `_check_labels_only_side` running **L1–L4** (labels.parquet exists/readable; label ∈
  {0,1}; utterance_id unique; count vs `--n-trials`) + **reused D6** (README front-matter) + **D7**
  (metric registered, via `loader._parse_eval_yaml` KeyError reuse) + **S1–S4** (submission checks
  unchanged → "same ✔ trust"). Skips audio checks D1/D3/D4/D5. Factor the submission loop into a shared
  helper.
- Tests (plan.md §D.4): `tests/test_validate_dataset_labels_only.py` + CLI test extension.

### 1C. Release
- Bump `pyproject.toml` + `__init__.py` version together (0.4.0 → **0.4.1**, additive: new metric +
  CLI flag, no schema change). Full `pytest` green; CLI smoke; both ≈50% smokes (eer + srr).
- Tag on `lab260ru/speech_spoof_bench`; bump `arena/requirements.txt` pin to the new SHA.

# Phase 2 — Arena: per-dataset metric rendering (minimal slice for Extended)

Because LRLspoof lands in **Extended**, the **Core** aggregate (`global_scores`/`assign_tiers`) is
untouched — no penalty/Platinum work needed now. Only the **per-dataset rendering** must resolve the
metric per dataset:
- `ranking.py`: in the per-dataset table path, resolve `metric_d = entry.get("metric", cfg["metric"])`
  and `lower_is_better` accordingly; add `_DEFAULT_PENALTY["srr_complement"] = 50.0` (plan.md RANK §F).
  Rank LRLspoof rows by the `srr_complement` score key (already present in the submission `scores`).
- `leaderboard.py`/`app.py`: render a **distinct column header** for spoof-only datasets
  ("Spoof-only · 1−SRR (%)") on the per-dataset tab (stage-5 §5). Cosmetic but clarifying.
- `ingest.py`: no change needed for scores (the `srr_complement` value rides in `Row.scores`); confirm
  the per-dataset tab reads `core_set + extended` (it does) so the Extended dataset gets a tab.
- Tests: per-dataset table ranks an Extended spoof-only dataset by `srr_complement`; header label.
- Verify exact current anchors before editing (`ranking.py:253 per_dataset_table`, leaderboard
  per-dataset path) — plan.md line anchors are the starting point.

# Phase 3 — Dataset: build + land in-place at `lab260/LRLspoof`

a. **Symlink** `benchmarks/LRLspoof -> /home/kirill/mnt/.../lrl_spoof` (user OK'd; read-only source).
b. **Enumerate** all `*.wav` (os.scandir, sorted) → `utterance_id` = relative POSIX path; build
   `data/labels.parquet` (`utterance_id`, `label=1`). Assert count; record `n_trials`.
c. **`eval.yaml`** (fork DeepVoice's): `name: LRLspoof`, spoof-only description incl. the utterance_id
   convention + DeepVoice-calibration note, `metrics: [srr_complement]`.
d. **README front-matter (in-place merge into lab260's card):** add `arena-ready` (+ `anti-spoofing`,
   `audio-deepfake-detection`, `speech`, `benchmark`) tags, `configs`, and the `arxiv` key
   (per decision #3). Keep the existing language list/license. Add a "Submitter workflow" section:
   download + extract the tarball → key `scores.txt` by relative path.
e. **Validate offline:** `speech-spoof-bench validate-dataset benchmarks/LRLspoof --labels-only
   --skip-submissions --n-trials <count>` → green (L1–L4 + D6 + D7).
f. **Push only the small files** to `lab260/LRLspoof` (proxy ON — tiny upload; the tarball is not
   touched): `data/labels.parquet`, `eval.yaml`, `README.md`, `submissions/random-baseline.yaml`.
   Capture the new SHA. Streaming sanity: `hf_hub_download` the labels parquet at the SHA, check count.
g. **Seed the random baseline** (via `submitting-arena-model` sub-skill): derive `t*` from
   `random-baseline`'s DeepVoice EER threshold; generate random scores over all utterance_ids; compute
   `srr_complement` (≈50%); write the v5 submission with the `calibration` block; open the submission PR
   into `lab260/LRLspoof/submissions/`.
h. **Manifest PR:** add the `extended:` entry + `srr_complement` to `metrics_in_use` + CHANGELOG
   `dataset_added`; validate locally; open PR; revert local clone.
i. **Land it (maintainer):** merge manifest PR (re-fetch main, confirm additive); `reproduce --scoring`
   the baseline locally (deterministic srr via calibration); fill `reproduction: {match: scoring,…}`;
   merge the submission PR; re-ingest (`ingest.load_state(force_refresh=True)`, row present, 0 warnings)
   + commit `cache.json` to the Space; verify live (Space RUNNING, `/healthz`, the LRLspoof tab/row).

---

# Execution log

Decisions at gate: **Core** placement; arxiv `2603.02364`; approved.

## Done + verified (local, reversible)
- [x] Phase 1: `srr_complement` + `call_metric` (explicit `wants_config`) + reproduce `_repro_config`
      + clean-FAIL guard; benchmark callsite routed; `validate-dataset --labels-only`/`--n-trials`.
      **Full suite 336 passed.** Version bumped 0.4.0 → **0.4.1** (`pyproject.toml` + `__init__.py`).
- [x] **Adversarial review** (3 agents): fixed 5 issues — fragile arity introspection → explicit
      `wants_config`; SRR rejects non-finite scores/threshold; labels-only rejects 0-row + null id.
      (+8 tests.)
- [x] Phase 2 (core): `ranking.global_scores` resolves per-dataset `metric` (+ `_metric_for`,
      `_penalty_for`, `_DEFAULT_PENALTY["srr_complement"]=50`). **arena ranking/leaderboard/overview
      68 passed.** Per-dataset tab already renders `srr_complement` via `per_dataset_table`.
- [x] Phase 3 (artifact): `benchmarks/LRLspoof/{eval.yaml,README.md,data/labels.parquet}` built from
      the local tree (NOT the 500 GB tarball). **n_trials = 1,304,455**, all distinct, all label=1,
      labels.parquet = 6.3 MB. `validate-dataset --labels-only` → **L1-L4+D6+D7 all green**.

## Remaining (OUTWARD / irreversible — gated on user go-ahead)
- [ ] Phase 2 polish: spoof-only column header ("Spoof-only · 1−SRR (%)") in overview UI (cosmetic).
- [ ] Push package 0.4.1 (commit+tag `lab260ru/speech_spoof_bench`) + bump `arena/requirements.txt` pin.
- [ ] Push arena files in-place to `lab260/LRLspoof` (labels.parquet + eval.yaml + README + submissions/);
      tarball untouched. NOTE: adds `configs`→labels.parquet (changes `load_dataset` behavior) + edits the live README.
- [ ] Seed random-baseline (t* from its DeepVoice submission → 1.3M random scores → srr≈50% + calibration block).
- [ ] Manifest PR: add `lab260/LRLspoof` to **core_set** (metric/category/penalty) + `srr_complement`
      to `metrics_in_use`; merge. **Board churn: ~11 full-coverage systems drop gold→silver (19/19→19/20).**
- [ ] Deploy arena code to the Space (per-dataset metric) + reproduce baseline + re-ingest + commit cache.json.
- [ ] Live confirm: Space RUNNING; LRLspoof tab shows the baseline ranked on 1−SRR.

Temp helper: `benchmarks/LRLspoof/_build_labels.py` (re-enumerates labels.parquet; safe to keep/delete).

## Notes / open risks
- **Cross-org submissions:** future external submissions PR into `lab260/LRLspoof/submissions/` — the
  arena webhook/CI is wired for the `SpeechAntiSpoofingBenchmarks` org; cross-org auto-routing is the
  known §7 hurdle. The baseline seed sidesteps it (maintainer reproduces + self-merges). Flag for later.
- **utterance_id ↔ tarball paths:** verify the tarball's internal paths match the local tree (cheap
  verification deferred — a tar part is ~11.6 GB so we won't download one; rely on the documented
  convention + the assumption that the tarball was built from this tree).
- **Scale:** ~1M+ files → enumeration is the slow step (background it); labels.parquet + scores.txt stay
  modest (~tens of MB).
