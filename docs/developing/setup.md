# Local Development Setup

The golden rule of this project: **prove every layer locally before it touches Hugging
Face or GitHub.** Most "it worked while I was developing" failures come from skipping the
offline loop and discovering a break only once it's live. This doc gets you a fast,
fully-offline inner loop.

## Prerequisites

- Python ≥ 3.10.
- A Hugging Face account + a token (`huggingface-cli login`, or set `HF_TOKEN`) — needed
  only for the *online* steps (fetching datasets, opening PRs). The offline loop needs no
  network.
- This user's environment uses **conda + a `uv` wrapper**. `pip` is intercepted by `uv`;
  install editable packages with the project's wrapper (see the `user_environment`
  memory). The commands below use plain `pip install -e .`; substitute your wrapper.

## Install the package editable

```bash
cd ~/speech-spoof-bench/speech-spoof-bench
pip install -e ".[dev]"        # dev extras = pytest, pytest-cov
speech-spoof-bench --help      # confirm the console script is on PATH
```

Editable install matters for two reasons:

1. Your code changes take effect without reinstalling — **except** packaged data
   (`schema/*.json`, the `dataset_skeleton/` template). Those are read via
   `importlib.resources`; if you edit a schema or the skeleton, the editable install
   usually still picks them up from `src/`, but if you see stale behaviour, reinstall.
2. `local_registry` writes `local-datasets.yaml` to the **repo root** (it resolves via
   `parents[2]` from the module), which only works from a source checkout.

## Run the test suite

```bash
pytest                          # ~all tests; uses fixtures, no network
pytest tests/metrics/test_eer.py -q
pytest tests/ci -q              # CI logic (HfApi / gh / subprocess are mocked)
```

The tests are the contract. `tests/fixtures/` holds known-good and known-bad submission
YAMLs and a `scores_known.txt` with a hand-checked EER. If you change a schema or the EER,
expect specific fixtures/tests to change with it — that's intentional (they're the
versioning contract; see [../architecture/versioning.md](../architecture/versioning.md)).

## The offline loop: register a dataset locally

Streaming a real dataset from HF on every iteration is slow. Register a local copy once:

```bash
# A local dataset dir must contain eval.yaml and data/test-*.parquet
speech-spoof-bench local set \
  SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA \
  /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA

speech-spoof-bench local list
speech-spoof-bench local show SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
```

Now any command that takes that `org/name` spec resolves to the local parquet instead of
streaming from HF — no network, instant. `local set` validates the directory has
`eval.yaml` + `data/test-*.parquet` before saving, so a wrong path fails immediately.

> The registry file is **gitignored**; the dataset repos live on a separate drive (see
> `workspace_layout` / `reference_repos` memory). `lookup()` raises `FileNotFoundError` if
> a registered path disappears — it never silently falls back to HF, so a stale entry
> fails loudly rather than quietly hitting the network. `unset` to clear it.

## Smoke-test with the random baseline

The package ships a reference model that scores N(0,1) — EER should land near 50% on a
balanced split. Use it to confirm your *environment* works before blaming your *model*:

```bash
speech-spoof-bench run \
  --model-module speech_spoof_bench.examples.random_baseline:RandomBaseline \
  --datasets SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA \
  --output-dir ./results
cat results/ASVspoof2019_LA/result.yaml     # eer_percent ≈ 50
```

If this works and your model doesn't, the problem is your model (or your bona-fide/spoof
score direction — see [new-model.md](new-model.md)).

## Confirm against canonical HF data

The local copy is a convenience; CI runs against the **canonical, pinned HF revision**.
Before you submit, run once with `--no-local` (and `--no-skip-existing` to force a fresh
run) to confirm there's no discrepancy between your local copy and the official snapshot:

```bash
speech-spoof-bench run --model-module my_model:MyModel \
  --datasets SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA \
  --no-local --no-skip-existing --output-dir ./results
```

## What lives where (quick orientation)

```
speech-spoof-bench/                  # the package repo (this one)
├── src/speech_spoof_bench/          # the code
│   ├── schema/*.json                # bundled JSON schemas (versioned contracts)
│   └── data/dataset_skeleton/       # the scaffold template (shipped in the wheel)
├── tests/                           # the contract; run before every change
├── results/                         # local run outputs (gitignored content)
├── local-datasets.yaml             # your offline registry (gitignored)
└── docs/                            # you are here
```

## Next

- Developing a model → [new-model.md](new-model.md)
- Building a dataset → [new-dataset.md](new-dataset.md)
- The full "what silently breaks" catalogue → [testing-and-pitfalls.md](testing-and-pitfalls.md)
</content>
