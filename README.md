# speech-spoof-bench

Benchmark harness for the [SpeechAntiSpoofingBenchmarks](https://huggingface.co/SpeechAntiSpoofingBenchmarks) org. Run anti-spoofing models against published datasets (HF) or a local copy on disk.

## Install

    pip install -e .

## Quick start

    speech-spoof-bench run \
        --model-module speech_spoof_bench.examples.random_baseline:RandomBaseline \
        --datasets /path/to/local/ASVspoof2019_LA \
        --output-dir ./results

For full design and roadmap see `docs/roadmap/PLAN.md` and `docs/roadmap/ROADMAP.md`.
