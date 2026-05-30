# The `speech_spoof_bench` Package

The pip package (`speech-spoof-bench` on PyPI-name, importable as `speech_spoof_bench`)
is the engine. This doc covers its evaluation core. The submission/validation side lives
in [submission-lifecycle.md](submission-lifecycle.md); the CI commands in
[cicd.md](cicd.md); badges in [badges.md](badges.md).

- **Version:** `0.1.0` (`pyproject.toml` and `__init__.py.__version__`).
- **Python:** `>=3.10`.
- **Layout:** `src/speech_spoof_bench/` (src layout; setuptools `packages.find` over `src`).
- **Entry point:** `speech-spoof-bench = "speech_spoof_bench.cli:main"`.
- **Runtime deps:** `datasets>=2.18`, `huggingface_hub>=0.20`, `numpy>=1.24`,
  `pyyaml>=6.0`, `jsonschema>=4.0`, `scipy>=1.10`.

> **Import note.** `__init__.py` only exports `__version__`. There is **no**
> `from speech_spoof_bench import Benchmark`. Import from the submodule:
> `from speech_spoof_bench.model import AntiSpoofingModel`,
> `from speech_spoof_bench.benchmark import Benchmark`.

## Module map

| Module | Responsibility |
|--------|----------------|
| `model.py` | The `AntiSpoofingModel` ABC and the `SimpleAntiSpoofingModel` convenience base. |
| `benchmark.py` | `Benchmark.run()` — the orchestrator. Writes `result.yaml`. |
| `runner.py` | Per-dataset loop: extract → resample → batch → score → write `scores.txt`. |
| `loader.py` | Resolve a dataset spec (local dir / HF `org/name`) to a streaming dataset + metadata. |
| `manifest.py` | Fetch + validate `arena-manifest`, expose Core/Extended dataset ids and revisions. |
| `hf_fetch.py` | Download a **commit-pinned** HF `resolve` URL and return its SHA-256. |
| `cache.py` | Purge the HF dataset cache between datasets to free disk. |
| `metrics/` | Pluggable metric registry; `eer.py` ships the launch metric `eer_percent`. |
| `local_registry.py` | Map `org/name` → local parquet dir for offline runs (`local-datasets.yaml`). |
| `examples/random_baseline.py` | Reference `SimpleAntiSpoofingModel` (N(0,1) scores → ~50% EER). |

## The model interface

This is the **one thing a contributor implements.** From `model.py`:

```python
class AntiSpoofingModel(abc.ABC):
    name: ClassVar[str] = "unnamed"
    expected_sample_rate: ClassVar[int] = 16000
    batch_size: ClassVar[int] = 1

    def load(self) -> None: ...          # called once, before any scoring
    @abc.abstractmethod
    def score_batch(self, audios: list[np.ndarray], srs: list[int]) -> list[float]: ...
    def unload(self) -> None: ...        # called once, at the end (in a finally)
```

Contract guarantees the runner gives you:

- Each `audio` is **float32, mono, resampled to `expected_sample_rate`** (16 kHz by default).
  You never resample yourself.
- **Higher score = more bona fide.** (Label `0` = bona fide, `1` = spoof.) This convention
  is what the EER metric assumes — getting it backwards inverts your EER (e.g. 2% becomes 98%).
- `score_batch` must handle any `k` with `1 <= k <= batch_size`; the runner sends the
  final short batch too.

For the common case there's `SimpleAntiSpoofingModel`: implement `score(audio, sr) -> float`
and you get `score_batch` for free (it just loops).

```python
from speech_spoof_bench.model import SimpleAntiSpoofingModel
import numpy as np

class MyModel(SimpleAntiSpoofingModel):
    name = "my-detector"
    expected_sample_rate = 16000
    batch_size = 8

    def load(self):
        self.net = load_checkpoint("checkpoint.pt")

    def score(self, audio: np.ndarray, sr: int) -> float:
        return float(self.net(audio).softmax(-1)[1])   # bona-fide logit

    def unload(self):
        del self.net
```

## Benchmark orchestration — `Benchmark.run()`

```python
Benchmark.run(
    model,                  # an AntiSpoofingModel instance
    datasets,               # list of specs: ["org/name", "/local/dir", ...]
    output_dir,             # e.g. "./results"
    streaming=True,
    cleanup=True,           # purge HF cache after each non-local dataset
    skip_existing=True,     # skip datasets whose result.yaml already matches the revision
    force_remote=False,     # ignore the local registry, always hit HF
) -> dict[str, BenchmarkResult]
```

Lifecycle:

1. `model.load()` **once**, before the dataset loop.
2. For each dataset spec → `loader.resolve()` → `runner.run_dataset()` (see below).
3. Load the resulting `scores.txt`, then for each metric id declared in the dataset's
   `eval.yaml`, look it up in the registry and compute it.
4. `_write_result_yaml()` writes `output_dir/<slug>/result.yaml`
   (`schema_version: 4`, with `artifact.scores_sha256` and
   `artifact.bench_version = "speech-spoof-bench==0.1.0"`).
5. If `cleanup` and the dataset is remote → `cache.purge_hf_cache()`.
6. `model.unload()` in a `finally` (runs even if a dataset blows up mid-loop).

`datasets="all"` (expand from the manifest) currently raises `NotImplementedError`.

## The per-dataset runner — `runner.py`

`run_dataset(model, source, dataset, output_dir)`:

- `_extract(row)` parses each HF row into `(utt_id, array, sr, label)`. The `utt_id` comes
  from the JSON in the row's `notes` column (`json.loads(notes)["utterance_id"]`).
- `_to_float32_mono_16k()` averages channels and resamples via
  `scipy.signal.resample_poly` (polyphase, using `gcd(sr, target_sr)`).
- Buffers utterances up to `model.batch_size`, then `_score_with_fallback()` calls
  `score_batch`. **If the batch raises, it retries each item individually** so one bad
  utterance doesn't lose the whole batch.
- Writes `scores.txt`: one `utt_id score\n` line per scored utterance.
- **`SKIP_FRACTION_THRESHOLD = 0.05`**: if more than 5% of items fail, it raises
  `TooManySkips` and aborts the dataset. (A handful of skips is fine and recorded as
  `n_skipped`.)

`scores.txt` example:
```
LA_E_1000001 0.123456
LA_E_1000002 -0.876543
```

## Dataset loading — `loader.py`

`resolve(spec, streaming=True, force_remote=False)` returns
`(DatasetSource, IterableDataset)`. Dispatch order:

1. `spec` is an existing directory → **local** (read `eval.yaml`, glob `data/test-*.parquet`).
2. `spec` looks like `org/name` **and** `local_registry.lookup()` finds it → **local**.
3. `spec` looks like `org/name` → **HF** (`load_dataset` with the `Audio` feature).
4. otherwise → `ValueError`.

`DatasetSource` (frozen dataclass) carries: `spec`, `display_name`, `slug`,
`canonical_id`, `metrics` (list of metric ids from `eval.yaml`), `split`, `is_local`,
`local_path`, `revision`.

Datasets stream as `IterableDataset` — **single-pass**. Iterating twice exhausts it; if
code needs a second pass it calls `resolve()` again.

## Metrics — `metrics/`

A tiny plugin registry. From `metrics/__init__.py`:

```python
@register_metric(id="eer_percent", display_name="EER (%)",
                 lower_is_better=True, requires_audio=False)
def compute_eer(scores: dict[str, float], labels: dict[str, int]) -> MetricResult: ...
```

- A metric is a `Callable[[ScoresMap, LabelsMap], MetricResult]`.
- `register_metric` records `id`, `display_name`, `lower_is_better`, `requires_audio`.
- **A metric only exists if its module is imported.** `metrics/__init__.py` does
  `from . import eer` so the decorator fires on package load. Add a new metric file *and*
  import it there, or `get_metric("your_id")` raises and `Benchmark.run` fails with
  "metric id not registered". → [developing/new-metric.md](../developing/new-metric.md)

### EER (`metrics/eer.py`)

Equal Error Rate, in percent, lower-is-better.

- `FAR(t) = P(spoof_score >= t)`, `FRR(t) = P(bonafide_score < t)`.
- Sweeps thresholds over the union of unique scores, finds the first sign-change of
  `FAR - FRR` (interpolating between thresholds); if there is no crossing, picks the
  threshold minimising `|FAR - FRR|`.
- Vectorised with `np.searchsorted` (O(N log N)).
- Returns the EER `value` (a percentage) plus `extras`: `threshold`, `n_trials`,
  `n_bonafide`, `n_spoof`.

## Manifest reader — `manifest.py`

`fetch_manifest()` downloads `manifest.yaml` from
`SpeechAntiSpoofingBenchmarks/arena-manifest` (a public HF dataset), parses it, and
validates against the **bundled** `schema/manifest.schema.json`. Accessors:
`core_dataset_ids()`, `all_dataset_ids()`, `revision_for(id)`. Used by `submit --datasets all`,
the `manifest`/`list` CLI commands, and the Arena. See the manifest shape in
[versioning.md](versioning.md).

## Reproducible downloads — `hf_fetch.py`

`parse_hf_resolve_url(url)` requires a `…/resolve/<sha>/…` URL where `<sha>` is 7–40 hex
chars (a **commit**, never a branch like `main`). `download(url)` fetches via
`hf_hub_download(revision=sha)`, honouring `$HF_TOKEN`, and returns
`(local_path, sha256_hex)`. This is what makes `reproduce --scoring` deterministic.

## The local registry — `local_registry.py`

Maps `org/name` → a local parquet directory so you can run fully offline (huge time saver
when iterating on a model, and what nightly checks use to avoid re-streaming). Persists to
`local-datasets.yaml` at the repo root (`schema_version: 1`). Key behaviours:

- `set(id, path)` validates the dir has `eval.yaml` + `data/test-*.parquet` before saving.
- `lookup(id)` returns the `Path` or `None`; if an id is registered but the path no longer
  exists it **raises `FileNotFoundError`** rather than silently falling back to HF — so a
  stale registry fails loudly. → [developing/setup.md](../developing/setup.md)

## The CLI surface (verified against `cli.py`)

```
speech-spoof-bench run            --model-module M:C --datasets SPEC [--datasets ...]
                                  [--output-dir DIR] [--no-streaming] [--no-cleanup]
                                  [--no-skip-existing] [--no-local]
speech-spoof-bench list
speech-spoof-bench manifest
speech-spoof-bench validate-dataset SPEC [--skip-submissions] [--no-local]
speech-spoof-bench validate-submission PATH
speech-spoof-bench submit         --model-module M:C --datasets SPEC [--datasets ...]
                                  --model-repo ORG/NAME --submission-meta PATH
                                  --hf-username NAME --contact EMAIL
                                  [--output-dir DIR] [--continue-on-error] [--no-local]
speech-spoof-bench scaffold-dataset --name NAME --output-dir DIR [--force]
speech-spoof-bench reproduce PATH (--scoring | --inference) [--tolerance F] [--no-local]
speech-spoof-bench local set DATASET_ID PATH
speech-spoof-bench local unset DATASET_ID
speech-spoof-bench local list
speech-spoof-bench local show DATASET_ID
speech-spoof-bench ci verify-pr   --repo ORG/NAME --pr N --branch REF
speech-spoof-bench ci nightly-revalidate [--open-issues]
speech-spoof-bench ci post-merge-badge --repo ORG/NAME --pr N --sha SHA
```

- `--model-module` is `module.path:ClassName`, e.g. `my_model:MyModel`.
- `--datasets` is repeatable; a single literal `all` expands from the manifest (in `submit`).
- `--no-local` (`force_remote=True`) bypasses the local registry — use it to confirm a
  run works against the canonical HF data, which is what CI does.
- `reproduce --inference` is **wired but raises `NotImplementedError`** (Phase 8+ feature).
</content>
