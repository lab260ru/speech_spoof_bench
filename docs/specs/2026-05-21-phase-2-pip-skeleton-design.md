# Phase 2 — Pip package skeleton (with local dataset loading)

**Status**: design
**Date**: 2026-05-21
**Scope**: Phase 2 of [ROADMAP.md](../roadmap/ROADMAP.md). Aligned with [PLAN.md §2](../roadmap/PLAN.md) (v4 spec).

## Goal

Ship the minimum pip package needed to run a model end-to-end against `ASVspoof2019_LA` and produce a `scores.txt` + partially-filled `result.yaml`. Plus: the runner must be able to load a dataset from a **local directory** (no HF download) when the user already has the parquet shards on disk — the LA copy at `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA` is the canonical test case.

Out of scope at this phase (deferred per ROADMAP): `submit`, `reproduce`, `ci verify-pr`, `scaffold-dataset`, webhook handler, full `validate-dataset` checks, manifest-driven dataset discovery.

## Package layout

```
speech-spoof-bench/
├── pyproject.toml
├── README.md
├── src/speech_spoof_bench/
│   ├── __init__.py
│   ├── model.py              # AntiSpoofingModel ABC + SimpleAntiSpoofingModel helper
│   ├── metrics/
│   │   ├── __init__.py       # register_metric, MetricResult, registry
│   │   └── eer.py            # eer_percent
│   ├── loader.py             # DatasetSource + resolve() — local-or-HF dispatch
│   ├── runner.py             # iterate rows, score batches, write scores.txt
│   ├── benchmark.py          # Benchmark.run orchestrator
│   ├── cache.py              # purge HF cache; no-op for local sources
│   ├── manifest.py           # stub at this phase
│   ├── cli.py                # argparse: run / list / validate-dataset
│   └── examples/
│       └── random_baseline.py
└── tests/
    ├── metrics/test_eer.py
    ├── test_loader.py
    └── test_runner.py
```

Sole deviation from PLAN.md §2.1: a new `loader.py` module to isolate the local-vs-HF dispatch from the runner. Everything else matches the spec.

## Module designs

### `model.py` — `AntiSpoofingModel`

```python
class AntiSpoofingModel(abc.ABC):
    name: str
    expected_sample_rate: int = 16000
    batch_size: int = 1   # runner passes this many items per score_batch call

    @abc.abstractmethod
    def load(self) -> None: ...

    @abc.abstractmethod
    def score_batch(self, audios: list[np.ndarray], srs: list[int]) -> list[float]:
        """One score per input. Higher = more bonafide. len(out) == len(audios)."""

    def unload(self) -> None: ...


class SimpleAntiSpoofingModel(AntiSpoofingModel):
    @abc.abstractmethod
    def score(self, audio: np.ndarray, sr: int) -> float: ...

    def score_batch(self, audios, srs):
        return [self.score(a, s) for a, s in zip(audios, srs)]
```

Contracts:
- Audio is float32 mono, resampled to 16 kHz by the runner before the model sees it.
- `load`/`unload` are called once per dataset.
- An exception inside `score_batch` aborts the whole batch; those items count as skipped.
- If `n_skipped / n_total > 0.05` for a dataset, the runner raises `TooManySkips`.
- Test-time augmentation: the model performs N internal augmentations inside `score_batch` and aggregates before returning one score per input. The runner sees one score per utterance regardless.
- `batch_size` lets each model declare its preferred chunk size — AASIST might set 16, the random baseline sets 1.

Rationale for dropping the misleading default `score_batch = [score(a) for ...]`: a no-op default looks like batching but isn't. Forcing implementers to write `score_batch` explicitly makes the batch contract honest. `SimpleAntiSpoofingModel` keeps the easy path easy.

### `metrics/__init__.py` — registry

```python
@dataclass(frozen=True)
class MetricResult:
    value: float
    extras: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class MetricSpec:
    id: str
    display_name: str
    lower_is_better: bool
    requires_audio: bool
    fn: Callable[[dict[str, float], dict[str, int]], MetricResult]

_REGISTRY: dict[str, MetricSpec] = {}

def register_metric(*, id, display_name, lower_is_better, requires_audio=False): ...
def get_metric(id: str) -> MetricSpec: ...
def list_metrics() -> list[MetricSpec]: ...
```

### `metrics/eer.py`

Registers `eer_percent`. Implementation: pool scores by label, sweep threshold for the FAR/FRR crossing, return percent EER. `extras` carries `threshold` and `n_trials`. Uses `scipy.interpolate` for the crossing.

### `loader.py` — local-or-HF dispatch

```python
@dataclass(frozen=True)
class DatasetSource:
    spec: str                    # original user input
    display_name: str            # eval.yaml `name:`
    slug: str                    # safe directory name, e.g. "ASVspoof2019_LA"
    canonical_id: str            # "<org>/<slug>" (HF) or "<slug>" (local)
    metrics: list[str]           # eval.yaml `tasks[0].metrics`
    split: str                   # eval.yaml `tasks[0].split`, default "test"
    is_local: bool
    local_path: Path | None
    revision: str | None         # commit sha (HF) or None (local)

def resolve(spec: str, *, streaming: bool = True) -> tuple[DatasetSource, IterableDataset]: ...
```

**Dispatch rule** — checked in order:
1. `Path(spec).is_dir()` → local mode.
2. `spec` matches `<org>/<name>` shape → HF mode.
3. Otherwise → raise `ValueError` with a clear message.

**Local mode:**
1. Read `<path>/eval.yaml`. Extract `name`, `tasks[0].metrics`, `tasks[0].split`.
2. `slug = Path(spec).name`.
3. `canonical_id = slug` (no org prefix; that's filled in at submit time in Phase 7).
4. `load_dataset("parquet", data_files=str(path / "data" / "test-*.parquet"), split="train", streaming=streaming)`. HF's parquet loader names the only split `"train"` — we relabel internally as the dataset's declared split. No network call.
5. `revision = None`.
6. Assert at least one parquet shard matches the glob; otherwise raise.

**HF mode:**
1. `load_dataset(spec, split=split, streaming=streaming)`.
2. `hf_hub_download(spec, "eval.yaml", repo_type="dataset")` to read display name + metrics.
3. `slug = spec.split("/")[-1]`, `canonical_id = spec`.
4. `revision = None` at Phase 2 (manifest resolution lands in Phase 4 — TODO comment in code).

**Validation** in both modes:
- `eval.yaml` parses; has non-empty `name`, `tasks[0].metrics`, `tasks[0].split`.
- Every metric id in `metrics:` exists in the registry; unknown ids raise immediately.

### `runner.py` — iterate + score

```python
@dataclass
class RunResult:
    scores_path: Path
    labels: dict[str, int]        # utt_id -> 0 (bonafide) | 1 (spoof)
    n_total: int
    n_skipped: int

def run_dataset(
    model: AntiSpoofingModel,
    source: DatasetSource,
    dataset: IterableDataset,
    output_dir: Path,
) -> RunResult: ...
```

Steps:
1. `output_dir.mkdir(parents=True, exist_ok=True)`. Open `scores.txt` for writing.
2. Iterate `dataset` in chunks of `model.batch_size`. For each chunk:
   - Extract `audio.array`, `audio.sampling_rate`, and `utterance_id` from `json.loads(row["notes"])`.
   - Resample to 16 kHz if needed (`scipy.signal.resample_poly`).
   - Call `model.score_batch(audios, srs)`.
   - On exception: log, count items as skipped, continue.
   - Write one line per item: `<utterance_id> <score>\n`.
3. Track `n_total`, `n_skipped`. If `n_skipped / n_total > 0.05`: raise `TooManySkips`.
4. Collect labels (0=bonafide, 1=spoof) for downstream metric computation.
5. Return `RunResult`.

The runner never touches the registry or writes `result.yaml`. That belongs to the orchestrator.

### `benchmark.py` — `Benchmark.run`

```python
def run(
    model: AntiSpoofingModel,
    datasets: list[str] | str = "all",
    output_dir: str | Path = "./results",
    streaming: bool = True,
    cleanup: bool = True,
    skip_existing: bool = True,
) -> dict[str, BenchmarkResult]: ...
```

Per dataset spec:
1. `source, ds = loader.resolve(spec, streaming=streaming)`.
2. `out = Path(output_dir) / source.slug`.
3. If `skip_existing` and `out/result.yaml` exists with matching `revision` → skip.
4. `model.load()`.
5. `run_result = runner.run_dataset(model, source, ds, out)`.
6. `model.unload()`.
7. For each metric id in `source.metrics`: `get_metric(id).fn(scores, run_result.labels)`; collect into a dict.
8. Write `out/result.yaml` — partially-filled v4 submission template (per PLAN.md §1.6):
   - **Populated**: `schema_version: 4`, `system.name`, `dataset.id` (= `source.canonical_id`), `dataset.revision` (or `null` in local mode), `dataset.split`, `scores.<metric_id>` for each computed metric, `scores.n_trials`, `scores.n_skipped`, `artifact.scores_sha256` (sha256 of `scores.txt`), `artifact.bench_version`.
   - **Left empty**: `system.description/code/checkpoint/paper`, `artifact.scores_url`, the entire `reproduction:` block, the entire `submitter:` block, `submitted_at`, `notes`. Submitter or the future `submit` CLI fills these in.

9. If `cleanup=True` and `source.is_local is False`: `cache.purge_hf_cache(source.canonical_id)`. No-op for local sources.

`datasets="all"` requires the manifest, which is Phase 4. At Phase 2, `"all"` raises `NotImplementedError` with a message pointing at Phase 4. Users must pass explicit specs.

### `cache.py`

One function:

```python
def purge_hf_cache(repo_id: str) -> None: ...
```

Uses `huggingface_hub.scan_cache_dir()` and selectively removes revisions for the given `repo_id`. No-op (early return) if no matching repo in the cache.

### `manifest.py`

Stub at Phase 2:

```python
def fetch_manifest() -> dict:
    raise NotImplementedError("manifest support lands in Phase 4")
```

File exists so the import surface is stable and CLI subcommands can wire in cleanly later.

### `cli.py`

`argparse`. Console script: `speech-spoof-bench = speech_spoof_bench.cli:main`.

```
speech-spoof-bench run \
    --model-module <module>:<ClassName> \
    --datasets <path-or-id> [--datasets <path-or-id> ...] \
    [--output-dir ./results] \
    [--no-streaming] [--no-cleanup] [--no-skip-existing]

speech-spoof-bench list                    # raises NotImplementedError at Phase 2
speech-spoof-bench validate-dataset <spec> # stub: loads, checks schema of first row
```

`--model-module` parses `module:ClassName`, imports, instantiates with no args, passes to `Benchmark.run`.
`--datasets` is repeatable; values pass through to `loader.resolve` so paths and HF ids work interchangeably.

`validate-dataset` (stub): calls `loader.resolve(spec)`, materializes the first row, verifies the schema is `{path, audio, label, notes}` and `notes` parses as JSON with a non-empty `utterance_id`. Full §1.9 checks land in Phase 7.

### `examples/random_baseline.py`

```python
class RandomBaseline(SimpleAntiSpoofingModel):
    name = "random-baseline"
    def load(self):    self.rng = np.random.default_rng(0)
    def unload(self):  self.rng = None
    def score(self, audio, sr): return float(self.rng.standard_normal())
```

Seeded so EER is reproducibly near 50%.

## Tests

- `tests/metrics/test_eer.py` — synthetic scores with analytically-known EER:
  - perfectly separable bonafide vs spoof → EER ≈ 0
  - fully-overlapping distributions → EER ≈ 50
  - one mid-range case checked against a reference computation
- `tests/test_loader.py`:
  - **Local**: build a temp dir with a tiny synthetic parquet (`data/test-00000-of-00001.parquet`) + hand-written `eval.yaml`; assert `DatasetSource` fields are correct, dataset iterates, schema matches.
  - **HF**: monkeypatch `datasets.load_dataset` + `hf_hub_download`; assert dispatch hits the HF branch with expected args.
  - **Bad input**: nonexistent path that doesn't look like a repo id → `ValueError`.
- `tests/test_runner.py` — synthetic `IterableDataset` of 10 items + dummy model; assert `scores.txt` is well-formed; inject one row where the model raises and assert `n_skipped == 1`.

No GPU, no network, no real datasets in CI. The Phase 6 smoke test exercises the real path manually.

## pyproject.toml

Core dependencies (per PLAN.md §2.8): `datasets`, `huggingface_hub`, `numpy`, `pyyaml`, `jsonschema`, `scipy`. Torch is NOT a runtime dep. Python `>=3.10`. Console script `speech-spoof-bench`.

## Done when

```bash
pip install -e .

speech-spoof-bench run \
    --model-module speech_spoof_bench.examples.random_baseline:RandomBaseline \
    --datasets /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA \
    --output-dir ./results
```

produces `./results/ASVspoof2019_LA/scores.txt` (~71k lines, one per utterance) and `./results/ASVspoof2019_LA/result.yaml` with `scores.eer_percent ≈ 50.0` and `dataset.id: ASVspoof2019_LA`. No HF download occurs. Re-running with `--skip-existing` (default) is a no-op.

Swapping `--datasets SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA` (HF id) yields equivalent output with `dataset.id: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA` and triggers an HF download (then cleans up under `--cleanup`).
