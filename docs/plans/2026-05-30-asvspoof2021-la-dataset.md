# ASVspoof2021_LA Dataset + random-baseline Rollout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. NOTE: this plan is **human-in-the-loop** — Claude authors every file; the **user runs every command that touches data or Hugging Face** and pastes back outputs (commit SHAs, EER, PR URLs), which Claude folds into the authored files.

**Goal:** Publish `SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA` as a Core arena dataset, then bring `random-baseline` to merged+verified on all Core datasets with badges and confirm it on the live Arena — auditing `new-dataset.md` for correctness along the way.

**Architecture:** Clone the existing `ASVspoof2021_DF` sibling repo shape. The build script is multiprocess (parallel probe + one worker process per shard) and probe-then-conditional (re-encode all clips only if a soundfile-decode probe trips; otherwise embed raw FLAC bytes). Validate offline → publish to HF → pin into `arena-manifest` core_set → run the baseline → submit/reproduce/merge → badge → confirm on Arena.

**Tech Stack:** Python, `datasets` (HF), `pyarrow`, `soundfile`, `librosa`, `huggingface_hub`/`huggingface-cli`, `speech-spoof-bench` CLI.

**Spec:** `docs/specs/2026-05-30-asvspoof2021-la-dataset-design.md`

**Reference sibling:** `benchmarks/ASVspoof2021_DF/` (read it side-by-side while executing).

**Source data:** `/home/kirill/mnt/users_4tb/datasets/ASVspoof2021_LA_eval/` — `flac/` (181,566 files, 16 kHz/16-bit), `trial_metadata.txt` (8 cols: `speaker uid codec transmission attack label trim phase`), `ASVspoof2021.LA.cm.eval.trl.txt`, `LICENSE.txt` (ODC-By). Counts: 18,452 bonafide / 163,114 spoof / 181,566 total.

---

## File Structure

New repo `benchmarks/ASVspoof2021_LA/`:

| File | Responsibility |
|---|---|
| `README.md` | HF dataset card front-matter (D6 keys + `arena-ready`) + body |
| `eval.yaml` | task + `metrics: [eer_percent]` |
| `LICENSE.txt` | ODC-By text (copied from source) |
| `build_parquet.py` | multiprocess probe-then-conditional raw→parquet build |
| `protocols/ASVspoof2021.LA.cm.eval.trl.txt` | official eval protocol (copied) |
| `submissions/README.md` | submitter instructions (DF copy, ids swapped) |
| `submissions/results_template.yaml` | schema-v4 template (DF copy, ids/n_trials swapped) |
| `tests/test_schema.py` | schema smoke test (DF copy, counts swapped) |
| `.gitignore`, `.gitattributes` | build artifacts ignore + parquet→LFS |
| `data/` | generated: `test-*.parquet` + `labels.parquet` |

Edited elsewhere:
- `arena-manifest/manifest.yaml` — add LA to `core_set` at the published SHA.
- `arena-manifest/CHANGELOG.yaml` — `dataset_added` note.
- Phase B: `benchmarks/ASVspoof2021_LA/submissions/random-baseline.yaml`, a `meta.yaml`, model-card badge edits.

---

## Phase A — The dataset

### Task A1: Scaffold the repo directory + copy static assets

**Files:**
- Create dir: `benchmarks/ASVspoof2021_LA/{protocols,submissions,tests,data}`
- Create: `.gitignore`, `.gitattributes`
- Copy: `LICENSE.txt`, `protocols/ASVspoof2021.LA.cm.eval.trl.txt`

- [ ] **Step 1: Create directories**

```bash
cd /home/kirill/speech-spoof-bench
mkdir -p benchmarks/ASVspoof2021_LA/{protocols,submissions,tests,data}
```

- [ ] **Step 2: Copy LICENSE + protocol from source**

```bash
SRC=/home/kirill/mnt/users_4tb/datasets/ASVspoof2021_LA_eval
DST=/home/kirill/speech-spoof-bench/benchmarks/ASVspoof2021_LA
cp "$SRC/LICENSE.txt" "$DST/LICENSE.txt"
cp "$SRC/ASVspoof2021.LA.cm.eval.trl.txt" "$DST/protocols/ASVspoof2021.LA.cm.eval.trl.txt"
```

- [ ] **Step 3: Write `.gitignore`** (Claude authors)

```
__pycache__/
.pytest_cache/
*.pyc
_clean_flac/
```

- [ ] **Step 4: Write `.gitattributes`** (Claude authors)

```
*.tar.gz filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
```

- [ ] **Step 5: Verify layout**

Run: `ls -R benchmarks/ASVspoof2021_LA`
Expected: the four dirs + `.gitignore`, `.gitattributes`, `LICENSE.txt`, `protocols/ASVspoof2021.LA.cm.eval.trl.txt`.

---

### Task A2: Author `eval.yaml`

**Files:** Create `benchmarks/ASVspoof2021_LA/eval.yaml`

- [ ] **Step 1: Write the file** (Claude authors — mirrors DF, LA name/description)

```yaml
name: ASVspoof 2021 LA
description: >
  Logical Access (LA) evaluation partition of ASVspoof 2021. Binary
  classification: bonafide vs. spoof, over telephony-codec / transmission
  conditions. EER computed on the official LA eval protocol.
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
      - eer_percent
```

- [ ] **Step 2: Sanity-check YAML parses**

Run: `python -c "import yaml;print(yaml.safe_load(open('benchmarks/ASVspoof2021_LA/eval.yaml'))['name'])"`
Expected: `ASVspoof 2021 LA`

---

### Task A3: Author `README.md` (HF card)

**Files:** Create `benchmarks/ASVspoof2021_LA/README.md`

- [ ] **Step 1: Write the file** (Claude authors). The re-encoding sentence in "License & redistribution" is **finalized in Task A7** once the build reveals clean vs. dirty; write the clean-path wording now and adjust later if needed.

````markdown
---
license: odc-by
language:
  - en
pretty_name: ASVspoof 2021 LA
task_categories:
  - audio-classification
size_categories:
  - 100K<n<1M
configs:
  - config_name: default
    data_files:
      - split: test
        path: "data/test-*.parquet"
tags:
  - anti-spoofing
  - audio-deepfake-detection
  - speech
  - benchmark
  - arena-ready
arxiv:
  - "2109.00537"
---

# ASVspoof 2021 LA

Benchmark-ready packaging of the **Logical Access (LA) evaluation partition** from ASVspoof 2021 for speech anti-spoofing and synthetic / deepfake voice detection.

## Overview

This dataset contains the LA evaluation subset of the ASVspoof 2021 challenge. The task is binary classification: **bonafide** (genuine human speech) vs. **spoof** (synthetic, converted, or replayed speech). The LA partition adds realistic telephony **codec / transmission** conditions on top of the 2019 LA attacks. The original dataset is available at https://www.asvspoof.org/index2021.html.

## License & redistribution

This dataset is redistributed under the **Open Data Commons Attribution License (ODC-By)**. See `LICENSE.txt` for the full text. The labels and evaluation protocol are unmodified. Audio is provided as canonical 16 kHz mono FLAC.

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `path` | `string` | Stable archive-relative path (e.g. `LA_E_9332881.flac`), unique within dataset |
| `audio` | `Audio(16000)` | Audio waveform, 16 kHz mono |
| `label` | `ClassLabel` | `"bonafide"` (index 0) or `"spoof"` (index 1) |
| `notes` | `string` | JSON with `utterance_id`, `speaker_id`, `codec`, `transmission`, `attack_id`, `trim`, `phase` |

`notes` example:
```json
{"utterance_id": "LA_E_9332881", "speaker_id": "LA_0009", "codec": "alaw", "transmission": "ita_tx", "attack_id": "A07", "trim": "notrim", "phase": "eval"}
```

## Quick Start

```python
from datasets import load_dataset

ds = load_dataset("SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA", split="test")
print(ds[0])
```

## Stats

| Stat | Value |
|------|-------|
| Total trials | 181,566 |
| Bonafide | 18,452 |
| Spoof | 163,114 |

Phase breakdown (in `notes.phase`): eval 148,176 · progress 16,464 · hidden 16,926.

## Source provenance

- Original dataset: https://www.asvspoof.org/index2021.html
- Evaluation protocol: `protocols/ASVspoof2021.LA.cm.eval.trl.txt`

## Evaluation

For evaluation instructions and submission format, see [`submissions/README.md`](submissions/README.md).

## Citation

**Original paper**: https://arxiv.org/abs/2109.00537

```bibtex
@inproceedings{yamagishi21_asvspoof,
  title     = {{ASVspoof 2021: accelerating progress in spoofed and deepfake speech detection}},
  author    = {Junichi Yamagishi and Xin Wang and Massimiliano Todisco and Md Sahidullah and Jose Patino and Andreas Nautsch and Xuechen Liu and Kong Aik Lee and Tomi Kinnunen and Nicholas Evans and Héctor Delgado},
  year      = {2021},
  booktitle = {{2021 Edition of the Automatic Speaker Verification and Spoofing Countermeasures Challenge}},
  pages     = {47--54},
  doi       = {10.21437/ASVSPOOF.2021-8},
}
```

## Maintainer

Contact: k.n.borodin@mtuci.ru
````

- [ ] **Step 2: Verify front-matter has all D6 keys**

Run:
```bash
python -c "
import yaml
fm = next(yaml.safe_load_all(open('benchmarks/ASVspoof2021_LA/README.md')))
req = {'license','language','pretty_name','task_categories','size_categories','configs','tags','arxiv'}
print('missing:', req - set(fm))
print('arena-ready:', 'arena-ready' in fm['tags'])
"
```
Expected: `missing: set()` and `arena-ready: True`.

---

### Task A4: Author `submissions/` files

**Files:** Create `submissions/README.md`, `submissions/results_template.yaml`

- [ ] **Step 1: Copy DF submissions README, swap dataset id + counts** (Claude authors)

Copy `benchmarks/ASVspoof2021_DF/submissions/README.md` verbatim, then replace every `ASVspoof2021_DF` → `ASVspoof2021_LA`. (No hardcoded `n_trials` in the README body besides paths.)

- [ ] **Step 2: Write `submissions/results_template.yaml`** (Claude authors — DF copy, id + n_trials swapped)

```yaml
schema_version: 4

system:
  name: ""
  slug: ""
  description: ""
  code: ""
  checkpoint: ""
  paper:
    arxiv_id: ""
    url: ""
    bibtex: |
      @article{...}

dataset:
  id: SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA
  revision: ""
  split: test

scores:
  eer_percent: 0.0
  n_trials: 181566
  n_skipped: 0

artifact:
  # Must be pinned by commit sha. Pattern:
  #   https://huggingface.co/<owner>/<repo>/resolve/<commit-sha>/.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA/scores.txt
  scores_url: ""
  scores_sha256: ""
  bench_version: ""

# Leave this block empty — the maintainer fills it in at merge.
reproduction:
  reproduced_by: ""
  reproduced_at: ""
  reproduced_bench_version: ""
  match: ""

submitter:
  hf_username: ""
  contact: ""

submitted_at: ""
notes: ""
```

- [ ] **Step 3: Verify the template parses and points at LA**

Run: `python -c "import yaml;d=yaml.safe_load(open('benchmarks/ASVspoof2021_LA/submissions/results_template.yaml'));print(d['dataset']['id'],d['scores']['n_trials'])"`
Expected: `SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA 181566`

---

### Task A5: Author `tests/test_schema.py`

**Files:** Create `benchmarks/ASVspoof2021_LA/tests/test_schema.py`

- [ ] **Step 1: Copy DF test, swap counts** (Claude authors)

Read `benchmarks/ASVspoof2021_DF/tests/test_schema.py`, copy it, and replace any hardcoded DF constants: `611829`→`181566`, `22617`→`18452`, `589212`→`163114`, and any `ASVspoof2021_DF`→`ASVspoof2021_LA` / `DF_E_`→`LA_E_` references. Keep the test logic identical.

- [ ] **Step 2: (defer run)** This test runs against built parquet — it executes in Task A7 after the sample build.

---

### Task A6: Author `build_parquet.py`

**Files:** Create `benchmarks/ASVspoof2021_LA/build_parquet.py`

- [ ] **Step 1: Write the full script** (Claude authors)

```python
"""Parquet build for the ASVspoof2021_LA HF dataset repo.

Reads the raw ASVspoof 2021 Logical Access (LA) eval flac files plus the
trial_metadata.txt key file and emits the canonical 4-column schema
(path / audio / label / notes) sharded into NUM_SHARDS parquet files.

Multiprocess + probe-then-conditional
-------------------------------------
A parallel probe samples source flac and tries to decode them with soundfile.
If any fail, the whole build takes the "dirty" path and re-encodes every clip
via librosa -> clean 16 kHz mono FLAC (PCM preserved bit-exactly). If the
sample is clean, the "clean" path embeds the raw source bytes directly (no
decode, fast).

Assembly runs one worker process per shard, so the build scales across cores.
Each shard is written atomically (temp + os.replace) and skipped if already
present, so a killed run resumes cheaply.

Sample mode (--limit N / env LA_BUILD_LIMIT): first N rows into a single shard,
skipping the full-count asserts -- used for the fast offline validate-dataset
pass.
"""

import argparse
import io
import json
import os
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import soundfile as sf
from datasets import Audio, ClassLabel, Dataset, Features, Value

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = Path("/home/kirill/mnt/users_4tb/datasets/ASVspoof2021_LA_eval")
FLAC_DIR = SRC_ROOT / "flac"
META_PATH = SRC_ROOT / "trial_metadata.txt"
PARQUET_DIR = REPO_ROOT / "data"
NUM_SHARDS = 24
EXPECTED_ROWS = 181566
EXPECTED_BONAFIDE = 18452
EXPECTED_SPOOF = 163114
TARGET_SR = 16000
PROBE_SAMPLE = 3000
WORKERS = int(os.environ.get("LA_BUILD_WORKERS", min(24, os.cpu_count() or 4)))

FEATURES = Features(
    {
        "path": Value("string"),
        "audio": Audio(sampling_rate=16000),
        "label": ClassLabel(names=["bonafide", "spoof"]),
        "notes": Value("string"),
    }
)


def parse_metadata():
    """Parse trial_metadata.txt (8 space-separated columns).

    Columns: speaker_id utterance_id codec transmission attack_id label trim phase
    """
    records = []
    with open(META_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(f"Expected 8 columns, got {len(parts)}: {line!r}")
            records.append(
                {
                    "speaker": parts[0],
                    "uid": parts[1],
                    "codec": parts[2],
                    "transmission": parts[3],
                    "attack": parts[4],
                    "label": parts[5],
                    "trim": parts[6],
                    "phase": parts[7],
                }
            )
    return records


def build_notes(rec):
    return json.dumps(
        {
            "utterance_id": rec["uid"],
            "speaker_id": rec["speaker"],
            "codec": rec["codec"],
            "transmission": rec["transmission"],
            "attack_id": rec["attack"],
            "trim": rec["trim"],
            "phase": rec["phase"],
        }
    )


def _probe_one(uid):
    try:
        sf.read(str(FLAC_DIR / f"{uid}.flac"))
        return None
    except Exception as e:  # noqa: BLE001
        return f"{uid}: {str(e).splitlines()[0][:100]}"


def probe_decodability(records):
    """Return True if the source flac needs re-encoding (any sample failed)."""
    rng = random.Random(0)
    sample = rng.sample(records, min(PROBE_SAMPLE, len(records)))
    uids = [r["uid"] for r in sample]
    failures = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for err in ex.map(_probe_one, uids, chunksize=32):
            if err:
                failures.append(err)
    print(f"Probe: {len(failures)}/{len(uids)} sample clips failed soundfile decode")
    for f in failures[:10]:
        print(f"  {f}")
    return len(failures) > 0


def _clip_duration(uid):
    info = sf.info(str(FLAC_DIR / f"{uid}.flac"))
    return info.frames / info.samplerate


def _ensure_long_first_row(records):
    """Swap a clip with duration >= 1.0s to index 0 (validator D3 checks row 0)."""
    for i in range(len(records)):
        if _clip_duration(records[i]["uid"]) >= 1.0:
            if i != 0:
                records[0], records[i] = records[i], records[0]
            return
    raise RuntimeError("No clip with duration >= 1.0s found")


def _audio_bytes(uid, dirty):
    """FLAC bytes for a uid.

    Clean path: raw source bytes (no decode). Dirty path: librosa-decoded +
    soundfile re-encoded clean FLAC (PCM preserved bit-exactly).
    """
    src = FLAC_DIR / f"{uid}.flac"
    if not dirty:
        return src.read_bytes()
    y, _ = librosa.load(str(src), sr=TARGET_SR, mono=True)
    buf = io.BytesIO()
    sf.write(buf, y, TARGET_SR, format="FLAC")
    return buf.getvalue()


def _build_shard(task):
    """Worker: build one shard parquet from its row slice. Resumable + atomic."""
    shard_index, rows, dirty, num_shards = task
    shard_name = f"test-{shard_index:05d}-of-{num_shards:05d}.parquet"
    final = PARQUET_DIR / shard_name
    if final.exists() and final.stat().st_size > 0:
        return (shard_index, len(rows), "skipped")

    def row_gen():
        for rec in rows:
            uid = rec["uid"]
            yield {
                "path": f"{uid}.flac",
                "audio": {"bytes": _audio_bytes(uid, dirty), "path": f"{uid}.flac"},
                "label": rec["label"],
                "notes": build_notes(rec),
            }

    with tempfile.TemporaryDirectory() as cache:
        ds = Dataset.from_generator(row_gen, features=FEATURES, cache_dir=cache)
        tmp = PARQUET_DIR / f".{shard_name}.tmp"
        ds.to_parquet(str(tmp))
        os.replace(tmp, final)
    return (shard_index, len(rows), "built")


def _partition(records, num_shards):
    n = len(records)
    per = (n + num_shards - 1) // num_shards
    out = []
    for i in range(num_shards):
        chunk = records[i * per : (i + 1) * per]
        if chunk:
            out.append(chunk)
    return out


def build():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    limit = args.limit
    if limit is None and os.environ.get("LA_BUILD_LIMIT"):
        limit = int(os.environ["LA_BUILD_LIMIT"])
    sample_mode = limit is not None

    print(f"Reading metadata from {META_PATH}")
    records = parse_metadata()
    print(f"Parsed {len(records)} rows")
    if not sample_mode:
        assert len(records) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS}, got {len(records)}"

    check = records if not sample_mode else records[: max(limit * 4, limit)]
    missing = [r["uid"] for r in check if not (FLAC_DIR / f"{r['uid']}.flac").exists()]
    assert not missing, f"{len(missing)} flac files missing, e.g. {missing[:5]}"

    records.sort(key=lambda r: r["uid"])
    _ensure_long_first_row(records)

    dirty = probe_decodability(records)
    print(f"Re-encode path: {'DIRTY (re-encode all)' if dirty else 'CLEAN (embed raw)'}")

    if sample_mode:
        records = records[:limit]
        num_shards = 1
        print(f"SAMPLE MODE: {len(records)} rows -> 1 shard")
    else:
        num_shards = NUM_SHARDS

    bona = sum(1 for r in records if r["label"] == "bonafide")
    spoof = sum(1 for r in records if r["label"] == "spoof")
    print(f"  bonafide={bona} spoof={spoof} total={len(records)}")

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    shards = _partition(records, num_shards)
    tasks = [(i, rows, dirty, num_shards) for i, rows in enumerate(shards)]
    print(f"Building {len(tasks)} shard(s) with {WORKERS} workers...")
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for idx, n, status in ex.map(_build_shard, tasks):
            print(f"  shard {idx}: {n} rows ({status})")

    _verify(num_shards, sample_mode)
    print("All verifications passed!")

    if not sample_mode:
        from speech_spoof_bench import labels

        out = labels.emit_labels(REPO_ROOT)
        print(f"Wrote {out}")


def _verify(num_shards, sample_mode):
    import pyarrow.parquet as pq

    shards = sorted(PARQUET_DIR.glob("test-*.parquet"))
    total = sum(pq.read_metadata(str(f)).num_rows for f in shards)
    uid_set, path_set, bona, spoof = set(), set(), 0, 0
    for f in shards:
        t = pq.read_table(str(f), columns=["path", "label", "notes"])
        for p, lab, n in zip(
            t.column("path").to_pylist(),
            t.column("label").to_pylist(),
            t.column("notes").to_pylist(),
        ):
            path_set.add(p)
            uid_set.add(json.loads(n)["utterance_id"])
            if lab == 0:
                bona += 1
            elif lab == 1:
                spoof += 1
    assert len(uid_set) == total, "Duplicate utterance_ids"
    assert len(path_set) == total, "Duplicate paths"
    if not sample_mode:
        assert total == EXPECTED_ROWS, f"{total} != {EXPECTED_ROWS}"
        assert bona == EXPECTED_BONAFIDE, f"bonafide {bona} != {EXPECTED_BONAFIDE}"
        assert spoof == EXPECTED_SPOOF, f"spoof {spoof} != {EXPECTED_SPOOF}"
    t0 = pq.read_table(str(shards[0]))
    assert set(t0.column_names) == {"path", "audio", "label", "notes"}, t0.column_names
    audio0 = t0.column("audio")[0].as_py()
    data, sr = sf.read(io.BytesIO(audio0["bytes"]))
    dur = len(data) / sr
    assert sr == 16000, f"row0 sr {sr} != 16000"
    assert dur >= 1.0, f"row0 dur {dur:.2f}s < 1.0s"
    print(f"  verify: {total} rows, row0 {sr}Hz {dur:.2f}s decodable OK")


if __name__ == "__main__":
    build()
```

- [ ] **Step 2: Byte-compile check** (Claude may run — no data touched)

Run: `python -m py_compile benchmarks/ASVspoof2021_LA/build_parquet.py && echo OK`
Expected: `OK`

---

### Task A7: Sample build + offline validation (USER runs; Claude folds back)

**Files:** generates `data/test-00000-of-00001.parquet` (sample).

- [ ] **Step 1: USER — sample build (50 rows)**

```bash
cd /home/kirill/speech-spoof-bench/benchmarks/ASVspoof2021_LA
LA_BUILD_LIMIT=50 python build_parquet.py
```
Expected tail: `All verifications passed!` and a printed `Re-encode path: CLEAN ...` or `DIRTY ...`. **Paste the `Re-encode path:` line back** — Claude finalizes the README re-encoding sentence (Task A3) accordingly.

- [ ] **Step 2: USER — offline validate (D1–D7)**

```bash
cd /home/kirill/speech-spoof-bench
speech-spoof-bench validate-dataset ./benchmarks/ASVspoof2021_LA --skip-submissions
```
Expected: all of D1–D7 green. Paste output; if any check is red, Claude fixes the relevant file and you re-run.

- [ ] **Step 3: USER — run the schema test**

```bash
cd /home/kirill/speech-spoof-bench/benchmarks/ASVspoof2021_LA
python -m pytest tests/test_schema.py -q
```
Expected: pass. Paste output.

- [ ] **Step 4: USER — clear the sample shard before the full build**

```bash
rm -f /home/kirill/speech-spoof-bench/benchmarks/ASVspoof2021_LA/data/test-00000-of-00001.parquet
```
(The full build writes `test-*-of-00024.parquet`; the leftover 1-shard file would otherwise pollute the glob.)

- [ ] **Step 5: Claude — finalize README re-encoding wording**

If Step 1 reported **DIRTY**, Claude updates the README "License & redistribution" paragraph to the DF-style wording: "…a fraction of the source FLAC files use an encoding the standard `libsndfile`/`soundfile` decoder cannot read, so each clip was decoded and re-encoded to a clean, universally-decodable FLAC; PCM samples are preserved bit-exactly (round-trip difference 0); the 16 kHz sampling rate is unchanged." If CLEAN, leave the current wording.

---

### Task A8: Full build (USER runs)

- [ ] **Step 1: USER — full multiprocess build**

```bash
cd /home/kirill/speech-spoof-bench/benchmarks/ASVspoof2021_LA
python build_parquet.py            # optionally LA_BUILD_WORKERS=N to tune
```
Expected tail: 24 shards built, `All verifications passed!`, `Wrote .../data/labels.parquet`. Resumable — re-run if interrupted (built shards are skipped). Paste the final summary.

- [ ] **Step 2: USER — local validate the full build (offline)**

```bash
cd /home/kirill/speech-spoof-bench
speech-spoof-bench validate-dataset ./benchmarks/ASVspoof2021_LA --skip-submissions
```
Expected: D1–D7 green against the 24-shard build. Paste output.

---

### Task A9: Publish to Hugging Face (USER runs; Claude folds back SHA)

- [ ] **Step 1: USER — create the HF dataset repo (idempotent)**

```bash
huggingface-cli repo create SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA --repo-type dataset -y
```
(Already-exists is fine.)

- [ ] **Step 2: USER — upload the whole folder (parquet auto-LFS)**

```bash
cd /home/kirill/speech-spoof-bench
huggingface-cli upload SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA \
  ./benchmarks/ASVspoof2021_LA . \
  --repo-type dataset \
  --commit-message "Add ASVspoof2021_LA canonical parquet build"
```
Paste the printed commit URL.

- [ ] **Step 3: USER — online validate against HF**

```bash
speech-spoof-bench validate-dataset SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA --skip-submissions
```
Expected: D1–D7 green against what HF serves. Paste output.

- [ ] **Step 4: USER — report the commit SHA**

```bash
python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA').sha)"
```
Paste the 40-char SHA — Claude uses it in Task A10.

---

### Task A10: Pin into the manifest core_set (Claude authors; USER pushes)

**Files:** Modify `arena-manifest/manifest.yaml`, `arena-manifest/CHANGELOG.yaml`

- [ ] **Step 1: Claude — add LA to `core_set`** at the reported SHA

In `arena-manifest/manifest.yaml`, append under `core_set`:
```yaml
  - id: SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA
    revision: <SHA-FROM-A9>
```
(`revision` must match `^[0-9a-f]{7,40}$`.)

- [ ] **Step 2: Claude — add a CHANGELOG note** (data change → NO schema/ranking bump)

In `arena-manifest/CHANGELOG.yaml` `events:`, append:
```yaml
  - {date: 2026-05-30, type: dataset_added, text: "ASVspoof2021_LA added to Core (181,566 LA eval trials)", dataset_id: "SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA"}
  - {date: 2026-05-30, type: note, text: "Re-ingest to subscribe ASVspoof2021_LA for webhook routing"}
```

- [ ] **Step 3: USER — push the manifest**

```bash
cd /home/kirill/speech-spoof-bench/arena-manifest
git add manifest.yaml CHANGELOG.yaml
git commit -m "data: add ASVspoof2021_LA to Core at <SHA>"
git push
```
Paste confirmation. The Space webhook should refresh the Arena cache on push.

- [ ] **Step 4: USER — confirm LA appears on the Arena**

Open the Arena (datasets/by-size tab) and confirm `ASVspoof2021 LA` is listed. If stale, trigger a re-ingest (per the DF rollout note). Paste a screenshot or confirmation.

---

## Phase B — random-baseline end-to-end

> `random-baseline` (slug `random-baseline`, model repo `SpeechAntiSpoofingBenchmarks/random-baseline-asas`) already has **opened** (un-merged, `reproduction: {}`) submissions on DF and 2019_LA. Target end state: merged+verified on **all three** Core datasets, badged, visible on the Arena. Do 2021_LA first; then chase DF + 2019_LA to merged.

### Task B1: Locate the baseline model module + meta

- [ ] **Step 1: Claude — find the baseline class** (reads package source; no data)

```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
grep -rn "random" src/speech_spoof_bench --include=*.py -l
speech-spoof-bench run --help
```
Resolve the `--model-module import.path:ClassName` (or a registered `--model random-baseline`) and record it here before proceeding.

- [ ] **Step 2: Claude — author `meta.yaml`** for the run (matches the existing DF/2019 submission `system` block)

```yaml
system:
  name: random-baseline
  slug: random-baseline
  description: >
    Reference random baseline. Returns N(0, 1) for every utterance using a
    fixed seed (seed=0). EER ≈ 50% by construction.
  code: https://github.com/SpeechAntiSpoofingBenchmarks/speech-spoof-bench
  checkpoint: https://huggingface.co/SpeechAntiSpoofingBenchmarks/random-baseline-asas
  params_millions: 1
  paper:
    arxiv_id: "1911.01601"
    url: https://arxiv.org/abs/1911.01601
    bibtex: |
      @article{wang2020asvspoof, title={ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech}, author={Wang, Xin and others}, journal={Computer Speech \& Language}, volume={64}, pages={101114}, year={2020}, publisher={Elsevier}}
notes: >
  Random baseline; paper field cites the ASVspoof dataset paper as a placeholder,
  consistent with the other random-baseline submissions. Reproduction left empty
  for the maintainer to fill at merge.
```

---

### Task B2: Run the baseline on 2021_LA (USER runs)

- [ ] **Step 1: USER — generate scores**

```bash
cd /home/kirill/speech-spoof-bench
speech-spoof-bench run \
  --model-module <RESOLVED-IN-B1> \
  --datasets SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA \
  --output-dir ./results
```
Expected: `results/random-baseline/scores.txt` + `result.yaml`; EER ≈ 50%; `n_skipped` ≈ 0 (must satisfy `len(scores)+n_skipped == 181566`). Paste the `result.yaml` (EER, n_trials, n_skipped, bench_version).

---

### Task B3: Submit the PR (USER runs)

- [ ] **Step 1: USER — submit (uploads pinned scores + opens PR)**

```bash
cd /home/kirill/speech-spoof-bench
speech-spoof-bench submit \
  --model-module <RESOLVED-IN-B1> \
  --datasets SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA \
  --model-repo SpeechAntiSpoofingBenchmarks/random-baseline-asas \
  --submission-meta ./meta.yaml \
  --hf-username SpeechAntiSpoofingBenchmarks \
  --contact k.n.borodin@mtuci.ru
```
Paste the printed PR URL and the local `submission.yaml` path.

---

### Task B4: Reproduce locally (USER runs — mirrors CI)

- [ ] **Step 1: USER — verify like CI before merging**

```bash
cd /home/kirill/speech-spoof-bench
speech-spoof-bench reproduce ./results/ASVspoof2021_LA/submission.yaml --scoring --no-local
```
Expected: SHA-256 OK, coverage OK (`len+skipped==181566`), recomputed EER matches claimed within 1e-6. Paste output. If it fails, consult the new-model.md failure table and fix before merge.

---

### Task B5: Merge + maintainer reproduction block (USER)

- [ ] **Step 1: USER — merge the PR** on HF, then fill the `reproduction:` block (or let the merge CI fill it). Confirm `submissions/random-baseline.yaml` on the dataset repo has a populated `reproduction:` (not `{}`).

- [ ] **Step 2: USER — confirm S-checks pass online**

```bash
speech-spoof-bench validate-dataset SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA
```
Expected: D1–D7 **and** S1–S4 green (no `--skip-submissions` now). Paste output.

---

### Task B6: Badges (Claude authors card edits; USER pushes)

- [ ] **Step 1: Claude — add the dynamic tier/rank badges to the model card**

Append to the `random-baseline-asas` model card (`README.md`):
```markdown
[![arena tier](https://img.shields.io/endpoint?url=https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/random-baseline/tier.json)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=random-baseline)
[![arena rank](https://img.shields.io/endpoint?url=https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/random-baseline/rank.json)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=random-baseline)
```

- [ ] **Step 2: USER — upload the static `result.yaml` projection** (the post-merge-badge CI normally posts this comment; if running manually, paste the merged submission and Claude produces the schema-v1 `result.yaml`, then:)

```bash
huggingface-cli upload SpeechAntiSpoofingBenchmarks/random-baseline-asas \
  result.yaml .eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA/result.yaml \
  --repo-type model --commit-message "Add ASVspoof2021_LA result projection"
```
Then push the updated model card. Paste confirmation.

---

### Task B7: Confirm on the Arena (USER)

- [ ] **Step 1: USER — verify the system row**

Open `…/SpeechAntiSpoofingArena?system=random-baseline` and confirm `random-baseline` shows with **ASVspoof2021_LA** coverage and a ~50% EER. Confirm the dynamic badges render (tier/rank). Paste confirmation/screenshot.

---

### Task B8: Bring DF + 2019_LA to merged+verified (USER, as needed)

- [ ] **Step 1: USER — for each of DF and 2019_LA**, if their `submissions/random-baseline.yaml` still has `reproduction: {}`: run `reproduce --scoring --no-local` on each, fill the reproduction block, merge, and confirm `validate-dataset <id>` is fully green (S1–S4). Paste status for both.

---

## Phase C — Doc audit

### Task C1: Audit `new-dataset.md` against what we actually did

**Files:** Modify `speech-spoof-bench/docs/developing/new-dataset.md` (only concrete inaccuracies)

- [ ] **Step 1: Claude — compile findings** while executing A1–B8. Check each claim against reality:
  - `scaffold-dataset` flags/output vs. what we used (we cloned DF instead — note if the doc implies scaffold is required).
  - `emit-labels` command name + behavior (`speech-spoof-bench emit-labels ./dir`).
  - `validate-dataset` `--skip-submissions` behavior and online vs. local guidance.
  - Manifest PR steps: `revision` regex, core vs. extended, `dataset_added` changelog note, no schema/ranking bump.
  - Any command name / path that has drifted.

- [ ] **Step 2: Claude — fix concrete errors inline** (no speculative rewrites). List each fix in the commit body.

- [ ] **Step 3: USER — push the doc fix**

```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
git add docs/developing/new-dataset.md
git commit -m "docs: correct new-dataset.md inaccuracies found during ASVspoof2021_LA rollout"
git push   # only if/when you want it on GitHub
```

---

## Self-Review (completed)

- **Spec coverage:** trial scope (all 181,566 — A6 parser keeps every row, no phase filter) ✓; Core placement (A10) ✓; probe-then-conditional + multiprocess (A6) ✓; 24 shards (A6 `NUM_SHARDS`) ✓; `odc-by` (A3) ✓; labels.parquet (A6 `emit_labels`) ✓; submissions files (A4) ✓; publish+pin (A9–A10) ✓; random-baseline on all Core + badges + arena (B1–B8) ✓; doc audit (C1) ✓.
- **Placeholders:** the only intentional `<…>` are runtime values the user supplies (the HF SHA in A10, the resolved model-module in B1, the PR URL in B3) — each has an explicit command that produces it. No "TBD"/"handle errors"/silent gaps.
- **Type/name consistency:** `EXPECTED_ROWS=181566` / `EXPECTED_BONAFIDE=18452` / `EXPECTED_SPOOF=163114`, `NUM_SHARDS=24`, `notes` keys, and the slug `random-baseline` are used identically across A6, A4, A7, B-tasks.
- **Known runtime dependency:** B1 must resolve the baseline `--model-module` before B2/B3 — flagged as the first B step.
