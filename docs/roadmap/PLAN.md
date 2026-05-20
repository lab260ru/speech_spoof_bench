# SpeechAntiSpoofingBenchmarks — Full Infrastructure Spec v3

This is the spec, not a roadmap. Every section is independently buildable. Cross-references use `§N`.

**v3 changes vs v2**:
- Only EER at launch. Metrics are a plugin registry — new metrics (e.g. min-tDCF) can be added later without touching dataset repos or the pip package's core.
- No `attack_id` anywhere. Schema is simpler.
- Tiers are generic and config-driven. New tiers can be added by editing the manifest.
- Only redistributable datasets are in scope. No loader-only repos.
- Maintainer-controlled verification: nothing lands in the arena until the maintainer reproduces the score locally. There is no "self-reported" tier.

---

## §0. Architecture overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  Org: huggingface.co/SpeechAntiSpoofingBenchmarks                    │
│                                                                      │
│  ┌─────────────────────┐    ┌──────────────────┐    ┌────────────┐   │
│  │ Dataset repos (N)   │◄───┤ arena-manifest   │───►│ Arena Space│   │
│  │ - parquet           │    │ (revisions,      │    │ (Gradio)   │   │
│  │ - submissions/*.yml │    │  metrics, tiers) │    │            │   │
│  │ - eval.yaml         │    └──────────────────┘    └────────────┘   │
│  └─────────────────────┘            ▲                      ▲         │
│           ▲                         │                      │         │
│           │ HF PR (submitter)       │ read on demand       │         │
│           │ → maintainer            │                      │         │
│           │   reproduces locally    │                      │         │
│           │   → merges              │                      │         │
│           │                         │                      │         │
│  ┌────────┴─────────────────────────┴──────────────────────┘         │
│  │  pip: speech-spoof-bench                                          │
│  │  - AntiSpoofingModel base class                                   │
│  │  - Benchmark.run: download → eval → score → cleanup               │
│  │  - Metric registry (EER at launch; pluggable)                     │
│  │  - CLI: run, validate, validate-dataset, reproduce                │
│  └────────────────────────────────────────────────────────────────────│
└──────────────────────────────────────────────────────────────────────┘
```

**Repos**:
1. `SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA` (dataset, exists)
2. `SpeechAntiSpoofingBenchmarks/<each-other-dataset>` (§1)
3. `SpeechAntiSpoofingBenchmarks/arena-manifest` (one YAML, §4)
4. `SpeechAntiSpoofingBenchmarks/arena` (Space, §3)
5. `github.com/SpeechAntiSpoofingBenchmarks/speech-spoof-bench` (pip package, §2)

**Conventions** (frozen):
- Dataset repo names: `<Source><Year>_<Partition>` — e.g. `ASVspoof2019_LA`. Real-world: `InTheWild`, `WaveFake`.
- Label classes: `bonafide`, `spoof`. Always. Index 0 = bonafide.
- Higher score = more bonafide. Always.

---

## §1. Dataset spec

Each dataset is built to **eventually qualify as an HF benchmark**, even before HF's allow-list approves it. Loadable parquet, valid `eval.yaml`, valid card YAML, citation/arxiv linkage, machine-readable submissions folder.

### §1.1 File layout

```
<dataset>/
├── README.md
├── LICENSE.txt
├── data/
│   └── test-NNNNN-of-NNNNN.parquet
├── protocols/
│   └── <upstream-protocol-file>
├── submissions/
│   ├── README.md
│   ├── results_template.yaml
│   └── <slug>.yaml
├── eval.yaml
└── build_parquet.py
```

### §1.2 Canonical schema

```python
Features({
    "path":  Value("string"),
    "audio": Audio(sampling_rate=16000),
    "label": ClassLabel(names=["bonafide", "spoof"]),
    "notes": Value("string"),
})
```

| Field | Required | Description |
|---|---|---|
| `path` | yes | Stable archive-relative path, unique within dataset. |
| `audio` | yes | 16 kHz mono. Resampled at build time. |
| `label` | yes | `["bonafide", "spoof"]`. Index 0 = bonafide. |
| `notes` | yes | JSON string. Must contain `utterance_id`. |

`notes` example:
```json
{"utterance_id":"LA_E_2834763","speaker_id":"LA_0039","subset":"eval"}
```

Rules:
- `notes` always parses with `json.loads`.
- `utterance_id` is unique and stable across re-shards.
- `notes` is informational; the scoring layer only needs `utterance_id`.

### §1.3 README structure

1. Title, one-line summary
2. YAML frontmatter (§1.4)
3. Overview
4. License & redistribution (must be redistributable — see §1.8)
5. Schema table (§1.2 verbatim)
6. Quick Start (`load_dataset` example)
7. Stats table (n_total, n_bonafide, n_spoof, total duration)
8. Source provenance
9. Evaluation (link to §2 + `submissions/README.md`)
10. Citation (§1.4)
11. Maintainer contact

### §1.4 README YAML frontmatter

```yaml
license: <upstream-license>
language: [en]
pretty_name: ASVspoof2019 LA
task_categories: [audio-classification]
size_categories: [10K<n<100K]
configs:
  - config_name: default
    data_files:
      - {split: test, path: "data/test-*.parquet"}
tags:
  - anti-spoofing
  - audio-deepfake-detection
  - speech
  - benchmark
  - arena-ready
paperswithcode_id: <if-applicable>
arxiv:
  - 1911.01601                     # HF indexes these
```

`arena-ready` is the marker the arena uses to discover datasets.

**Citation block (required in README body)** — original paper's arXiv link + BibTeX:

```markdown
## Citation

**Original paper**: https://arxiv.org/abs/1911.01601

```bibtex
@article{wang2020asvspoof, ... }
```
```

If a dataset has a second canonical paper, include both.

### §1.5 `eval.yaml`

One task per dataset. Metric list is **explicit** so adding metrics later is a one-line change.

```yaml
name: ASVspoof 2019 LA
description: >
  Logical Access partition of ASVspoof 2019. Binary classification:
  bonafide vs. spoof. Metrics computed on the official LA eval protocol.
evaluation_framework: inspect-ai

tasks:
  - id: antispoofing_eval
    config: default
    split: test

    field_spec:
      input: audio
      target: label

    solvers:
      - name: speech_spoof_bench_solver

    scorers:
      - name: speech_spoof_scorer

    metrics:
      - eer_percent                  # launch metric
      # Future: add metric ids here; arena picks them up automatically.
```

`metrics` is a list of metric ids that must exist in the pip package's metric registry (§2.6). Datasets opt into metrics they support — a dataset that lacks the metadata for a given metric simply omits its id.

### §1.6 Submission format

`submissions/<system-slug>.yaml`. Every field is verifiable from the artifact bundle.

```yaml
schema_version: 1

system:
  name: AASIST
  slug: aasist-clovaai-default
  paper: https://arxiv.org/abs/2110.01200
  code: https://github.com/clovaai/aasist
  checkpoint: https://huggingface.co/...
  description: Reference AASIST, default config.

dataset:
  id: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
  revision: 7f3a9b1c                       # required
  split: test

scores:
  eer_percent: 0.83
  n_trials: 71237
  n_skipped: 0
  # Additional metric ids are added here as they're rolled out.

artifact:
  scores_file: scores/aasist-clovaai-default.txt
  scores_sha256: 4f9b...
  bench_version: speech-spoof-bench==0.3.1

reproduction:
  reproduced_by: SpeechAntiSpoofingBenchmarks      # required, see §1.7
  reproduced_at: 2026-05-20
  reproduced_bench_version: speech-spoof-bench==0.3.1
  match: scoring                                    # "scoring" or "inference"

submitter:
  hf_username: kborodin
  contact: k.n.borodin@mtuci.ru

submitted_at: 2026-05-19
notes: "Reference implementation, default config, FP32."
```

**Scores file** (`submissions/scores/<slug>.txt`):
```
LA_E_2834763 -1.234
LA_E_1665632  2.871
```
One line per utterance. Higher = more bonafide.

### §1.7 Verification (maintainer-controlled, mandatory)

**Nothing lands in the arena until a maintainer reproduces it locally.** There is no self-reported tier.

Two verification levels — every merged submission has at least the first:

| Level | What's checked | Cost |
|---|---|---|
| `scoring` | Maintainer re-runs `speech-spoof-bench reproduce --scoring` on the submitted `scores.txt`. Computes EER from it. Must match the YAML's `scores.eer_percent` exactly (within 1e-6). | Seconds. No audio, no GPU. |
| `inference` | Maintainer re-runs the full pipeline (`speech-spoof-bench reproduce --inference`): downloads the system's checkpoint, runs it on the dataset, regenerates `scores.txt`, recomputes EER. Must match within 0.05% EER. | Expensive. Per-system, per-dataset. |

Workflow:
1. Submitter opens an HF PR with `<slug>.yaml` + `scores/<slug>.txt`.
2. Maintainer runs `speech-spoof-bench reproduce --scoring <PR-branch>` — fast, cheap. If it fails, comment + reject.
3. If passes, maintainer writes the `reproduction:` block (level = `scoring`) and merges.
4. Later, the maintainer can run `--inference` on selected submissions and upgrade the level in a follow-up PR.

The `reproduction:` block being present is what the arena treats as "verified." A submission without it never appears in the arena (§3.3).

### §1.8 Redistribution

**In scope**: only datasets we can redistribute under their original license. No loader-only repos. No proxying.

If we can't legally rehost the audio, the dataset is out of scope for the org. This keeps every dataset self-contained, reproducible, and trivially loadable.

### §1.9 Validation (no CI)

Validation is a local command in the pip package (§2.5):

```bash
speech-spoof-bench validate-dataset SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
```

Checks:
- Schema is exactly `{path, audio, label, notes}`.
- `label` classes are `["bonafide", "spoof"]`.
- `notes` parses as JSON for a sample of 100 rows; every parsed object has a non-empty `utterance_id`.
- All `utterance_id` and `path` values are unique (full-dataset scan).
- `audio` sampling rate is 16000; first row decodes ≥ 1 s.
- README frontmatter has all keys from §1.4 including `arxiv`.
- `eval.yaml` parses, matches §1.5 shape, every metric id is registered in the pip package.
- Submission YAMLs in `submissions/` parse against the bundled JSON Schema, scores SHA matches the file, `reproduction:` block is present.

Run by: dataset maintainer before pushing; arena maintainer before adding to manifest; submitter on their own bundle before opening a PR.

### §1.10 Dataset DoD (per dataset)

- [ ] `speech-spoof-bench validate-dataset <id>` exits 0.
- [ ] Redistributable under upstream license; verbatim `LICENSE.txt` shipped.
- [ ] `n_bonafide + n_spoof == n_total`, matches README stats table.
- [ ] At least 1 submission in `submissions/` with `reproduction:` filled in at launch.
- [ ] `arena-ready` tag and `arxiv:` list present in YAML frontmatter.
- [ ] Citation section in README with arXiv link + BibTeX.

---

## §2. Pip package `speech-spoof-bench`

One package, does everything: download, run model, score, emit submission YAML, validate, reproduce.

### §2.1 Package layout

```
speech-spoof-bench/
├── pyproject.toml
├── README.md
├── src/speech_spoof_bench/
│   ├── __init__.py
│   ├── model.py                # AntiSpoofingModel
│   ├── benchmark.py            # Benchmark.run
│   ├── runner.py
│   ├── metrics/
│   │   ├── __init__.py         # registry: register/lookup by id
│   │   ├── eer.py              # eer_percent
│   │   └── README.md           # how to add a metric
│   ├── manifest.py
│   ├── submission.py
│   ├── validate.py
│   ├── reproduce.py            # maintainer-side scoring/inference reproduction
│   ├── cache.py
│   ├── inspect_integration.py
│   ├── cli.py
│   ├── schema/
│   │   ├── submission.schema.json
│   │   └── manifest.schema.json
│   └── examples/
│       ├── aasist.py
│       ├── rawnet2.py
│       └── random_baseline.py
└── tests/
```

### §2.2 `AntiSpoofingModel` base class

```python
class AntiSpoofingModel(abc.ABC):
    name: str
    expected_sample_rate: int = 16000

    @abc.abstractmethod
    def load(self) -> None: ...

    @abc.abstractmethod
    def score(self, audio: np.ndarray, sr: int) -> float:
        """Higher = more bonafide."""

    def score_batch(self, audios, srs) -> list[float]:
        return [self.score(a, s) for a, s in zip(audios, srs)]

    def unload(self) -> None: ...
```

Contract: audio is float32 mono, already resampled. `load`/`unload` called once per dataset. `score` exception on a single utterance → skipped; >5% skips aborts that dataset.

### §2.3 `Benchmark.run`

```python
def run(
    model: AntiSpoofingModel,
    datasets: list[str] | str = "all",
    output_dir: str | Path = "./results",
    streaming: bool = True,
    cleanup: bool = True,
    batch_size: int = 1,
    skip_existing: bool = True,
) -> dict[str, BenchmarkResult]: ...
```

### §2.4 Runner

1. Resolve `revision` from `arena-manifest`.
2. `load_dataset(id, split="test", streaming=streaming, revision=revision)`.
3. `model.load()`.
4. Iterate rows → `model.score_batch` → write `scores.txt`.
5. `model.unload()`.
6. For each metric id listed in the dataset's `eval.yaml`: look up in registry (§2.6), call it.
7. Emit `result.yaml` (= submission template, missing `reproduction:` block).
8. If `cleanup=True`: purge HF cache for that dataset.

Resumability: if `result.yaml` exists and its `revision` matches the manifest → skip.

### §2.5 CLI

```bash
speech-spoof-bench run --model-module mymodel:MyModel --datasets all
speech-spoof-bench list
speech-spoof-bench validate <submission.yaml>
speech-spoof-bench validate-dataset <repo-id>
speech-spoof-bench manifest

# Maintainer-only commands:
speech-spoof-bench reproduce --scoring <pr-branch-or-yaml>
speech-spoof-bench reproduce --inference <pr-branch-or-yaml>
```

### §2.6 Metric registry (the only path for adding new metrics)

Metrics are plugin-style. The package ships `eer_percent`. Adding a new metric is one file:

```python
# src/speech_spoof_bench/metrics/<id>.py
from speech_spoof_bench.metrics import register_metric, MetricResult

@register_metric(
    id="eer_percent",
    display_name="EER (%)",
    lower_is_better=True,
    requires_audio=False,            # most metrics only need scores.txt + labels
)
def compute(scores: dict[str, float], labels: dict[str, int]) -> MetricResult:
    """
    scores: utterance_id -> bonafide score (higher = bonafide)
    labels: utterance_id -> 0 (bonafide) or 1 (spoof)
    """
    ...
    return MetricResult(value=eer_pct, extras={"threshold": ..., "n_trials": ...})
```

Adding e.g. min-tDCF later:
1. Drop `metrics/tdcf.py` registering `min_tdcf` (and any per-dataset config it needs).
2. Bump the package version.
3. Add `min_tdcf` to the `metrics:` list in the `eval.yaml` of datasets that support it.
4. Existing submissions stay valid — they just don't have the new metric.
5. New submissions include it automatically.

The arena reads whatever metric ids it finds in submissions and displays them as columns; no arena code change needed.

### §2.7 Inspect-AI integration (optional install)

`pip install speech-spoof-bench[inspect]` registers the solver/scorer used by `eval.yaml`. Dormant until HF allow-lists the framework.

### §2.8 Versioning

SemVer. Public API: `AntiSpoofingModel`, `Benchmark.run`, metric registration decorator, CLI flags. Adding a metric is a minor bump, not major.

Core install: `datasets`, `huggingface_hub`, `numpy`, `pyyaml`, `jsonschema`, `scipy`. Torch only via `[examples]`.

### §2.9 Package DoD

- [ ] `pip install speech-spoof-bench` works, no GPU deps required.
- [ ] `python -m speech_spoof_bench.examples.random_baseline` runs end-to-end on `ASVspoof2019_LA`.
- [ ] AASIST example reproduces seeded baseline within 0.05% EER.
- [ ] EER on the official ASVspoof LA baseline scores file matches the published number within 0.01%.
- [ ] `cleanup=True` leaves no files outside `output_dir`.
- [ ] Mid-run kill → restart only re-runs the unfinished dataset.
- [ ] `validate-dataset` catches: wrong sample rate, missing `utterance_id`, duplicate `path`, missing `reproduction:` block in submissions.
- [ ] `reproduce --scoring` exits non-zero when the YAML's EER doesn't match what the scores file produces.
- [ ] Adding a new metric file under `metrics/` is sufficient to make it usable everywhere (covered by a registry test).
- [ ] CI matrix: Python 3.10/3.11/3.12 on Linux + macOS.

---

## §3. Arena Space

### §3.1 Stack

- Gradio, free CPU Space.
- Python 3.11.
- Lives at `SpeechAntiSpoofingBenchmarks/arena`. PR review through HF discussions.

```
arena/
├── README.md
├── app.py
├── ingest.py
├── ranking.py
├── schema.py
└── tests/
```

### §3.2 Tabs

1. **Overview** — tiered global ranking with mean rank, coverage, per-Core EER.
2. **Per dataset** — dropdown → per-system table with all available metrics as columns.
3. **System detail** — one page per system slug.
4. **Submit** — instructions, copy-paste commands, link to template.
5. **About** — schema version, ranking version, metrics list, last-refreshed timestamp.

### §3.3 Data flow (simplified — no GitHub Action)

1. On Space startup: fetch manifest; for each `arena-ready` dataset, list `submissions/*.yaml` via HF API; skip any submission missing a `reproduction:` block (§1.7); validate the rest; build in-memory table; cache 30 min.
2. Refresh button on every tab clears the cache and re-fetches.

Why this is enough:
- HF Spaces auto-restart often.
- Submissions are merged by maintainers (a manual action) — maintainer hits Refresh after merging.
- Numbers are at most ~24h stale, which is fine for a research leaderboard.

Upgrade path (not needed at launch): a small JSON cache file the Space commits back to itself, or — much later — a GitHub Action (free for public repos, 2000 min/month).

### §3.4 In-memory row shape

```python
@dataclass
class Row:
    system_slug: str
    dataset_id: str
    revision: str
    scores: dict[str, float]      # metric_id -> value (any registered metric)
    reproduction_level: str       # "scoring" | "inference"
    submitted_at: str
    submission_url: str
```

### §3.5 Verification badges (display only)

Two badges, both real, both maintainer-set:

| Badge | Meaning |
|---|---|
| ✔ scoring | Maintainer reproduced EER from the submitted `scores.txt`. Default for every merged row. |
| ★ inference | Maintainer also re-ran the model end-to-end and reproduced the scores within tolerance. |

No "self-reported" badge — those don't exist in the arena.

### §3.6 Generic tier system

Tiers are **defined in the manifest**, not hardcoded. Each tier has a `min_coverage` threshold (fraction of Core Set covered). New tiers added later by editing the manifest and bumping `ranking_version`.

Manifest excerpt (§4):
```yaml
tiers:
  - {name: gold,     min_coverage: 1.0}
  - {name: silver,   min_coverage: 0.5}
  - {name: bronze,   min_coverage: 0.0}
```

Adding e.g. `platinum` later is just inserting a row above gold with `min_coverage: 1.0` and a tie-breaker rule (e.g. "all submissions at `inference` verification level"). The arena reads the tier list and renders one table per tier in order.

### §3.7 Ranking logic

```python
def assign_tiers(submissions, tiers, core_set):
    """
    For each system_slug:
      coverage = |core covered| / |core|
      tier = highest tier whose min_coverage <= coverage and any extra
             conditions (e.g. min_verification_level) are satisfied
    Within a tier:
      Per Core dataset, per primary-metric: rank systems that submitted.
      Mean rank = mean over the (dataset, metric) cells the system has.
    Tie-break: more covered → mean rank on Extended → earlier date.
    """
```

Per dataset, the **primary metric** for ranking is the first id in that dataset's `eval.yaml` `metrics:` list. At launch every dataset has `eer_percent` as primary. When new metrics roll out, each dataset chooses whether to keep EER primary or switch.

Missing data is never imputed.

### §3.8 Arena DoD

- [ ] Cold-starts in <15s with seeded data.
- [ ] Refresh button works; new submissions appear within seconds.
- [ ] Submission missing `reproduction:` is skipped with a logged warning in About; doesn't break the Space.
- [ ] Tier rendering is driven by the manifest — adding a new tier in the manifest produces a new table in the Overview without code changes.
- [ ] Per-dataset tab dynamically renders columns from whichever metric ids are present.
- [ ] Ranking unit tests cover: full coverage, partial coverage, single submission, ties.
- [ ] "Last refreshed" timestamp visible on every tab footer.

---

## §4. Manifest repo (`arena-manifest`)

One file: `manifest.yaml`. Versioned by git tags.

```yaml
ranking_version: v1
schema_version: 1

metrics_in_use:                  # informational; controls arena's column order
  - eer_percent

tiers:                           # ordered, highest first
  - {name: gold,     min_coverage: 1.0}
  - {name: silver,   min_coverage: 0.5}
  - {name: bronze,   min_coverage: 0.0}

core_set:
  - id: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
    revision: 7f3a9b1c
  - id: SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA
    revision: ...
  - id: SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF
    revision: ...
  - id: SpeechAntiSpoofingBenchmarks/InTheWild
    revision: ...

extended:
  - id: SpeechAntiSpoofingBenchmarks/ASVspoof2019_PA
    revision: ...
```

DoD:
- [ ] Pip package reads without auth.
- [ ] Arena reads without auth.
- [ ] Bumping `ranking_version` is a tagged release with CHANGELOG.
- [ ] Adding a tier or metric is a tagged release; existing data continues to work.

---

## §5. Scalability — does this work on a free CPU HF Space?

Short answer: **yes, easily.** The arena does not run models, does not load audio, does not store audio. Hard numbers below.

### §5.1 What lives where

| Data | Where | Size implication for arena |
|---|---|---|
| Audio (FLAC/WAV in parquet) | Dataset repos on HF (~10–50 GB each) | Zero — arena never touches it |
| Submission YAML | Dataset repos, `submissions/*.yaml` | ~2 KB each |
| `scores.txt` | Dataset repos, `submissions/scores/*.txt` | ~2 MB each (one number per utterance) |
| Manifest | `arena-manifest` | <10 KB |
| Arena in-memory table | Space RAM | KB range, see §5.2 |

### §5.2 Arena footprint at scale (20 datasets × 30 systems)

The arena loads YAML metadata only — **not scores.txt files**. Scores files are linked for download in the UI but not parsed by the arena.

- 20 datasets × 30 systems = 600 submission YAMLs to fetch.
- ~2 KB each → 1.2 MB total YAML download per cold start.
- Parsed in-memory table: 600 rows × ~500 bytes = 300 KB.
- HF API list calls: 20 (one `list_repo_files` per dataset) + 600 file fetches. With HTTP keepalive and parallelism, ~10–20 s cold start. Comfortably under any Space restart budget.

Free CPU Spaces: 2 vCPU, 16 GB RAM, 50 GB disk. The arena uses ~100 MB resident. No bottleneck.

### §5.3 Where the real disk cost lives — and how the pip package handles it

The disk cost is in **running the benchmark**, not in the arena.

- ASVspoof 2019 LA eval: ~10 GB FLAC.
- ASVspoof 2021 LA + DF: ~30–80 GB combined.
- ITW: ~10 GB.
- 20 redistributable benchmarks: probably 100–300 GB total in the org's dataset repos.

HF dataset repo storage is effectively unlimited for public repos, so storing it is free. The cost is **downloading it to evaluate**.

The pip package handles this with `cleanup=True` (default): exactly **one dataset is on local disk at a time**. After scoring, the HF cache for that dataset is purged before downloading the next. Peak local disk for a 20-dataset run on a contributor's machine: the size of the single largest dataset (~50 GB worst case), not the sum.

With `streaming=True` (default), audio is iterated row-by-row from a streaming parquet reader — you don't even need the full dataset on disk, just whatever's in the streaming buffer. The cleanup step then removes the partial cache. In practice this brings the working footprint down to a few GB on contributor machines.

### §5.4 Verification cost on the maintainer side

This is the part that has real compute cost — and it's not on the Space, it's on whoever is the maintainer.

- **Scoring verification** (§1.7 level 1, mandatory): cheap. Re-reads `scores.txt` (~2 MB) and dataset `notes` for labels (streamed, ~few MB), recomputes EER. Seconds. No GPU. Free CPU machine is plenty.
- **Inference verification** (§1.7 level 2, optional): expensive. Equivalent to running the submitter's model on the dataset yourself. For one system × one dataset on a single GPU, this is minutes-to-hours depending on model + dataset size. Across 20 datasets × 30 systems × inference verification = 600 runs — that's where you'd want HF Jobs or a real GPU. Not at launch.

Scoring-verification scales fine. Mandatory verification at the cheap level keeps everything truthful without breaking the maintainer's budget.

### §5.5 Conclusion

The arena Space is comfortably within free CPU resources for the foreseeable future, and scales with no architectural changes to 20 datasets × 30 systems. The pip package's design (streaming + cleanup) keeps contributor disk use bounded to one dataset at a time. The expensive part — inference verification — is opt-in, per-submission, and not on the Space at all.

---

## §6. Cross-cutting

### §6.1 JSON Schemas (single source: pip package)

- `submission.schema.json` — `submissions/*.yaml`; requires `reproduction:` block.
- `manifest.schema.json` — manifest; requires `tiers` (ordered) and `core_set`.

### §6.2 Versioning matrix

| Artifact | Versioning |
|---|---|
| Dataset content | Git revision sha (pinned in manifest) |
| Dataset schema | `schema_version` in submission YAML |
| Manifest | SemVer tag |
| Ranking rules / tiers / metrics | `ranking_version` |
| `speech-spoof-bench` | SemVer; metrics added as minor bumps |
| Arena code | Untagged main |

### §6.3 Validation surface

| Command | Who runs it |
|---|---|
| `validate-dataset <id>` | Dataset maintainer; arena maintainer |
| `validate <submission.yaml>` | Submitter |
| `reproduce --scoring` | Maintainer, mandatory before merge |
| `reproduce --inference` | Maintainer, optional, upgrades badge |

### §6.4 Security & abuse

- Submissions are HF PRs.
- Scores files are inert text. Maintainer-only commands are the only thing that executes submitter-supplied code (`reproduce --inference`); maintainer decides when to run it.
- Reject submissions whose `scores_sha256` doesn't match.

### §6.5 Documentation

1. Org README — what the org is, Core Set, contribution flow.
2. `submissions/README.md` — one page, copy-pasted into each dataset.
3. `speech-spoof-bench` README — install, model wrapping, CLI.

---

## §7. Build order

| # | Milestone | Output |
|---|---|---|
| M1 | Schema migration on `ASVspoof2019_LA` (§1.2) + README citation/arxiv + `eval.yaml` per §1.5 | Updated dataset repo |
| M2 | `speech-spoof-bench` v0.1: `AntiSpoofingModel`, runner, metric registry with `eer_percent`, `validate-dataset`, `reproduce --scoring`, random + AASIST examples | pip package |
| M3 | Seed 3 baselines on LA; maintainer reproduces each; merge submissions with `reproduction:` filled in | populated `submissions/` |
| M4 | `arena-manifest` with LA pinned, tiers defined | manifest |
| M5 | Arena MVP — generic tier rendering, dynamic metric columns | Space live |
| M6 | Add `InTheWild` — validates schema generalizes | second dataset |
| M7 | Add `ASVspoof2021_LA`, `ASVspoof2021_DF` | Core complete |
| M8 | Add `reproduce --inference`; selectively upgrade top entries to ★ badge | trust signal |
| M9 | Extended: PA, WaveFake, ASVspoof5 | breadth |

Stop-the-line rules:
- If `validate-dataset` fails on the second dataset (M6), the schema is wrong — revisit §1.2.
- If `reproduce --scoring` disagrees with a submitter's claimed EER for any seeded baseline, fix before M3 merges.
- If the arena renders incorrect ranks on a hand-crafted test fixture, don't launch M5.

---

## §8. Open decisions

1. **One vs. two tasks in `eval.yaml`** when more metrics arrive. (Lean: one task; the metrics list under it is the extension point.)
2. **Inference verification for Gold tier**: should Gold eventually require ★ on every Core entry, or stay `scoring`-level? (Lean: stay `scoring`; add a separate "Verified Gold" tier once we have the compute budget.)
