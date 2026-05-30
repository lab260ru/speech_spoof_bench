# Developing a New Dataset

The **developer** companion to [submit-dataset.md](../submitting/submit-dataset.md). The
goal here is to get `validate-dataset` to **all-green offline** before anything is pushed
to Hugging Face, because a dataset that fails D1–D7 will silently never appear on the
Arena (ingest skips unverifiable submissions and the manifest won't accept it).

## The canonical shape

Every dataset is an HF *dataset* repo with this exact structure:

```
<name>/
├── README.md            # YAML front-matter = HF card; must tag `arena-ready`
├── eval.yaml            # task + metrics; metrics[0] is the PRIMARY metric
├── LICENSE.txt
├── build_parquet.py     # YOUR script: raw audio → canonical parquet
├── data/
│   └── test-*.parquet   # the four-column canonical schema
└── submissions/
    ├── README.md
    └── results_template.yaml
```

Canonical parquet columns (nothing more, nothing less):

| Column | Type | Notes |
|--------|------|-------|
| `path` | `string` | original relative path; must be unique |
| `audio` | `Audio(sampling_rate=16000)` | mono, ≥ 1.0 s |
| `label` | `ClassLabel(names=["bonafide", "spoof"])` | order matters: bonafide=0, spoof=1 |
| `notes` | `string` | JSON with a unique `utterance_id` (and anything else you like) |

## Step 1 — Scaffold

```bash
speech-spoof-bench scaffold-dataset --name MyDataset --output-dir ./mydataset
```

This materialises the template, substituting `{{NAME}}` in `README.md` and `eval.yaml`.
You get `README.md`, `eval.yaml`, `LICENSE.txt`, a `build_parquet.py` stub, and
`submissions/{README.md,results_template.yaml}`. The template ships inside the package, so
if you want to *change the template itself* you edit
`src/speech_spoof_bench/data/dataset_skeleton/` and reinstall (that's a package change —
see [contributing-package.md](contributing-package.md)).

## Step 2 — Redistribution check (do this first)

Before writing any code, confirm the source licence permits redistribution of the audio.
The project does **not** accept loader-only repos — the parquet must contain the actual
audio, legally. If you can't redistribute, stop here.

## Step 3 — Build the parquet

Fill in `build_parquet.py` to emit `data/test-*.parquet` with the four columns. The
critical, easy-to-get-wrong bits:

```python
from datasets import Dataset, Features, Audio, ClassLabel, Value
import json

features = Features({
    "path":  Value("string"),
    "audio": Audio(sampling_rate=16000),          # resamples on write
    "label": ClassLabel(names=["bonafide", "spoof"]),
    "notes": Value("string"),
})

def rows():
    for i, (wav_path, is_spoof) in enumerate(catalogue):
        yield {
            "path": wav_path,
            "audio": wav_path,                      # datasets reads & encodes the file
            "label": "spoof" if is_spoof else "bonafide",
            "notes": json.dumps({"utterance_id": f"MY_{i:07d}"}),  # MUST be unique
        }

ds = Dataset.from_generator(rows, features=features)
ds.to_parquet("data/test-00000-of-00001.parquet")
```

- **`utterance_id` must be unique and stable.** It's the join key for scoring. If you
  re-shard later and the ids change, every submission's coverage check breaks (D5 / the
  reproduce coverage check).
- **Resample to 16 kHz** (the `Audio(sampling_rate=16000)` feature handles the resample on
  write). D3 only spot-checks the **first row's** sample rate and duration, so ensure
  **row 0 is ≥ 1.0 s** — it does *not* scan every clip. Whole-set minimum duration is the
  builder's responsibility; drop sub-second clips yourself if your task requires it.
- Shard files must match `data/test-*.parquet` — the loader globs exactly that.

## Step 4 — README front-matter & eval.yaml

The README front-matter (HF card) **must** contain every D6 key and the `arena-ready` tag:

```yaml
---
license: <spdx-id>
language: [en]
pretty_name: "My Dataset"
task_categories: [audio-classification]
size_categories: [10K<n<100K]
configs:
  - config_name: default
    data_files: [{split: test, path: data/test-*.parquet}]
tags: [anti-spoofing, audio-deepfake-detection, speech, benchmark, arena-ready]
arxiv: <id-or-omit>
---
```

`eval.yaml` declares the task and the metric list; **`metrics[0]` is the primary metric**
the Arena ranks and badges by:

```yaml
name: MyDataset
description: "..."
evaluation_framework: inspect-ai
task:
  id: antispoofing_eval
  config: default
  split: test
  field_spec: {input: audio, target: label}
  solvers: [{name: speech_spoof_bench_solver}]
  scorers: [{name: speech_spoof_scorer}]
  metrics: [eer_percent]       # every id here MUST be registered in the package (D7)
```

If you list a metric the installed package doesn't know, D7 fails. Adding a metric is a
package change → [new-metric.md](new-metric.md).

## Step 5 — Validate until all-green (the whole point)

Run the validator against your **local directory** and fix every red check before you push:

```bash
speech-spoof-bench validate-dataset ./mydataset
```

The D-checks, and what trips them:

| Check | Fails when | Fix |
|-------|-----------|-----|
| D1 columns | extra/missing/renamed column | emit exactly `{path, audio, label, notes}` |
| D2 labels | label isn't `ClassLabel[bonafide, spoof]` | use the `ClassLabel` feature, that order |
| D3 audio | **first row's** sr≠16000 or duration <1.0 s | resample on write; ensure row 0 ≥ 1.0 s |
| D4 notes | a row's `notes` isn't JSON / lacks `utterance_id` | every row needs valid JSON with the key |
| D5 uniqueness | duplicate `utterance_id` or `path` | make ids/paths unique |
| D6 README | missing front-matter key or no `arena-ready` tag | add the keys + tag |
| D7 metrics | `eval.yaml` lists an unregistered metric | register it / use `eer_percent` |

Use `--skip-submissions` while you have no submissions yet (S1–S4 need a real PR/scores).

> Validate the **local dir** while iterating (fast, offline). Validate the **HF
> `org/name`** once after pushing to confirm it works against what HF actually serves.

## Step 6 — Publish, then open a manifest PR

1. Push the dataset repo to `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/<name>`.
2. `speech-spoof-bench validate-dataset SpeechAntiSpoofingBenchmarks/<name>` (online) — confirm green.
3. Note the **commit SHA**.
4. Open a PR on **`arena-manifest`** adding the dataset to `core_set` (or `extended`) with
   that pinned SHA:

```yaml
core_set:
  - id: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
    revision: 9b2040e8c57749dcd9a4f16ad61b4f47626b89ec
  - id: SpeechAntiSpoofingBenchmarks/MyDataset      # ← new
    revision: 1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b
```

- **Core vs Extended:** Core datasets count toward coverage/tiers and the global ranking;
  Extended are shown but don't gate tiers. Adding to Core changes everyone's coverage —
  that's a deliberate, reviewed decision.
- Adding a dataset is a **data** change → no `schema_version`/`ranking_version` bump. Add
  a `CHANGELOG.yaml` ➕ `dataset_added` note so it shows on the Arena's Over-time tab.
- The revision must be **lowercase hex, 7–40 chars** (`^[0-9a-f]{7,40}$`).

## Later: re-pinning a dataset

If you fix labels or re-shard, commit the dataset and **update the `revision` in the
manifest** to the new SHA (+ a ↻ `dataset_repin` changelog note). Old submissions stay
reproducible against the old revision; new runs use the new one. Re-pinning Core can move
scores, so call it out. See [../architecture/versioning.md](../architecture/versioning.md).
</content>
