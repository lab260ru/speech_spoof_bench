---
license: TODO
language: [en]
pretty_name: {{NAME}}
task_categories: [audio-classification]
size_categories: [unknown]
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
paperswithcode_id:
arxiv: []
---

# {{NAME}}

TODO: one-line summary.

## Overview

TODO: longer description, source, motivation.

## License & redistribution

TODO: confirm the upstream license permits redistribution and paste it
verbatim into `LICENSE.txt`. If not redistributable, this dataset is out of
scope for the org (PLAN.md §1.8).

## Schema

| Field | Type | Description |
|---|---|---|
| path  | string | Stable archive-relative path, unique within dataset. |
| audio | Audio(16kHz mono) | Resampled at build time. |
| label | ClassLabel[bonafide, spoof] | Index 0 = bonafide. |
| notes | string (JSON) | Must contain `utterance_id`. |

## Quick Start

```python
from datasets import load_dataset
ds = load_dataset("SpeechAntiSpoofingBenchmarks/{{NAME}}", split="test")
```

## Stats

| n_total | n_bonafide | n_spoof | total duration |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

## Source provenance

TODO

## Evaluation

See `eval.yaml` and `submissions/README.md`.

## Citation

**Original paper**: TODO arxiv link

```bibtex
TODO
```

## Maintainer

TODO
