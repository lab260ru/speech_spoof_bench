# speech-spoof-bench

A command-line tool and small Python library that scores speech **anti-spoofing**
(audio deepfake-detection) models against published datasets (each frozen to one exact
version) — and feeds the results to the live
[**Speech Anti-Spoofing Arena**](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena)
leaderboard.

## What it is

- A **CLI** (`speech-spoof-bench`) that runs your model over a dataset and writes a
  `scores.txt` plus a `result.yaml` with the computed score (EER — Equal Error Rate,
  where lower is better).
- A small **library**: you wrap your model in one tiny Python class, and the toolkit does
  the rest — streaming the audio, scoring each clip, computing the metric, and opening the
  submission pull request.

Every number it produces can be recomputed from the model's own score file, frozen to one
exact commit so it can never be quietly changed after the fact. That is what keeps the
Arena leaderboard honest: nothing is taken on the submitter's word.

## Install

```bash
# Install the latest from GitHub (it isn't on PyPI yet):
pip install git+https://github.com/lab260ru/speech_spoof_bench.git

# Or, from a clone of this repo, to develop it:
pip install -e ".[dev]"               # adds pytest
```

Requires Python 3.10+.

## 30-second sanity check

The package ships a random-guessing reference model. Run it to confirm your *environment*
works before you blame your *model* — a random model should land near **50% EER**:

```bash
speech-spoof-bench run \
  --model-module speech_spoof_bench.examples.random_baseline:RandomBaseline \
  --datasets SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA \
  --output-dir ./results
cat results/ASVspoof2019_LA/result.yaml     # eer_percent ≈ 50
```

This is a quick does-it-run check (a *smoke test*), not a real submission. If it works but
your own model scores badly, the problem is your model — most often a flipped score sign
(see the [FAQ](docs/FAQ.md)).

## Submit your model or dataset

- **[Submit a model](docs/submitting/submit-model.md)** — wrap your model, score a
  dataset, open a pull request, appear on the leaderboard.
- **[Submit a dataset](docs/submitting/submit-dataset.md)** — package a new benchmark for
  everyone to score against.

## Develop & extend

- **[Documentation map](docs/README.md)** — start here to understand the whole system.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — pick your task; *adding a metric* is a good
  first PR.
- **[FAQ](docs/FAQ.md)** — the handful of things that silently trip people up.
- **[Testing & pitfalls](docs/developing/testing-and-pitfalls.md)** — what breaks in
  production even when it passes on your machine.

## The live leaderboard

The results feed the
**[Speech Anti-Spoofing Arena](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena)**
— an open, reproducible leaderboard where every row has been re-checked, not trusted on
faith.

## How the project is laid out

This package (the CLI + library) lives on GitHub under **`lab260ru`**. The datasets, the
leaderboard Space, and the ranking manifest all live on Hugging Face under the
**`SpeechAntiSpoofingBenchmarks`** org. The [documentation map](docs/README.md) explains
how the pieces fit together.

---

*Project history & original design: [docs/roadmap/ROADMAP.md](docs/roadmap/ROADMAP.md)
and [docs/roadmap/PLAN.md](docs/roadmap/PLAN.md).*
