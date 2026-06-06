# ASVspoof5 Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the ASVspoof5 track_1 eval split (680,774 clips) as an Arena-ready HF dataset under `benchmarks/ASVspoof5/`, validate it all-green, push it to the HF org, and add it to the manifest's Core set.

**Architecture:** Fork the proven `benchmarks/ASVspoof2021_LA/` packaging. The source FLAC is already clean 16 kHz mono (full 680,774-clip decode probe: 0 failures), so the builder embeds raw source bytes directly — no decode/re-encode. Records are processed sorted by `utterance_id`, which both maximizes HDD read throughput (~440 vs ~65 files/s) and yields stable sharding. Output: 200 parquet shards (~420 MB each, ~80 GB total) plus `data/labels.parquet`.

**Tech Stack:** Python, `datasets`, `soundfile`, `pyarrow`, `huggingface_hub`, `speech-spoof-bench` CLI.

---

## Reference facts (verified, not assumptions)

- Source root (read-only): `/home/kirill/mnt/users_4tb/datasets/asvspoof5/last_eval`
  - `flac_E_eval/<utterance_id>.flac` — 681,872 files, 80 GB; all 16 kHz mono, clips 4–10 s.
  - `ASVspoof5.eval.track_1.tsv` — 680,774 rows, 10 whitespace-separated columns.
- Protocol columns (1-indexed): `1 speaker_id · 2 utterance_id(=flac name) · 3 gender(M/F) · 4 codec(C0x or "-") · 5 codec_id(0–5) · 6 source_id · 7 attack_condition(AC1/AC2/AC3 or "-") · 8 attack_id(bonafide or Axx) · 9 label(bonafide/spoof) · 10 "-" (constant, dropped)`.
- Counts: total **680,774**, bonafide **138,688**, spoof **542,086**.
- Device: `/dev/sdc`, spinning HDD. Fastest read = sorted filename order, 64 workers.
- HF: logged in as `korallll`; org `SpeechAntiSpoofingBenchmarks`; manifest repo `SpeechAntiSpoofingBenchmarks/arena-manifest`.

## File structure (created under `benchmarks/ASVspoof5/`)

| File | Responsibility |
|------|----------------|
| `build_parquet.py` | Parse protocol → emit `data/test-*.parquet` (raw-byte embed) + verify |
| `eval.yaml` | Task + primary metric (`eer_percent`) |
| `README.md` | HF card front-matter (`arena-ready`, ODC-By) + stats/provenance/citation |
| `LICENSE.txt` | ODC-By v1.0 (copied verbatim from ASVspoof2021_LA) |
| `.gitattributes` | `*.parquet` → LFS |
| `.gitignore` | caches |
| `submissions/README.md`, `submissions/results_template.yaml` | submission scaffold |
| `data/test-*.parquet`, `data/labels.parquet` | build outputs (not committed to the package git; pushed to HF) |

---

## Task 1: Scaffold the dataset repo skeleton

**Files:**
- Create: `benchmarks/ASVspoof5/` (via CLI)
- Copy: `benchmarks/ASVspoof5/LICENSE.txt`

- [ ] **Step 1: Scaffold**

Run:
```bash
cd /home/kirill/speech-spoof-bench/benchmarks
speech-spoof-bench scaffold-dataset --name ASVspoof5 --output-dir ./ASVspoof5
```
Expected: creates `ASVspoof5/{README.md,eval.yaml,LICENSE.txt,build_parquet.py,submissions/...}`.

- [ ] **Step 2: Use the same ODC-By LICENSE as the sibling datasets**

Run:
```bash
cp /home/kirill/speech-spoof-bench/benchmarks/ASVspoof2021_LA/LICENSE.txt \
   /home/kirill/speech-spoof-bench/benchmarks/ASVspoof5/LICENSE.txt
cp /home/kirill/speech-spoof-bench/benchmarks/ASVspoof2021_LA/.gitattributes \
   /home/kirill/speech-spoof-bench/benchmarks/ASVspoof5/.gitattributes
cp /home/kirill/speech-spoof-bench/benchmarks/ASVspoof2021_LA/.gitignore \
   /home/kirill/speech-spoof-bench/benchmarks/ASVspoof5/.gitignore
```
Expected: files copied.

- [ ] **Step 3: Verify scaffold**

Run: `ls -la /home/kirill/speech-spoof-bench/benchmarks/ASVspoof5`
Expected: `README.md eval.yaml LICENSE.txt build_parquet.py .gitattributes submissions/`.

---

## Task 2: Write `build_parquet.py`

**Files:**
- Modify (overwrite stub): `benchmarks/ASVspoof5/build_parquet.py`

- [ ] **Step 1: Write the builder**

Overwrite `benchmarks/ASVspoof5/build_parquet.py` with:

```python
"""Parquet build for the ASVspoof5 (track_1 eval) HF dataset repo.

Reads the ASVspoof5 eval FLAC + the track_1 protocol TSV and emits the canonical
4-column schema (path / audio / label / notes) sharded into NUM_SHARDS parquet
files. A full decode probe of all 680,774 clips found 0 soundfile failures, so
this builder takes the CLEAN path: it embeds the raw source FLAC bytes directly
(no decode/re-encode). A small probe still runs as a safety gate and aborts if
the source has regressed.

Records are processed sorted by utterance_id: on the source HDD this turns random
seeks (~65 files/s) into near-sequential reads (~440 files/s) and gives stable
sharding.

Sample mode (--limit N): first N rows into a single shard, skipping the
full-count asserts -- used for the fast offline validate-dataset pass.
"""

import argparse
import io
import json
import os
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import datasets  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import soundfile as sf  # noqa: E402
from datasets import Audio, ClassLabel, Dataset, Features, Value  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

try:
    datasets.disable_progress_bars()
except AttributeError:
    from datasets.utils.logging import disable_progress_bar

    disable_progress_bar()

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = Path("/home/kirill/mnt/users_4tb/datasets/asvspoof5/last_eval")
FLAC_DIR = SRC_ROOT / "flac_E_eval"
META_PATH = SRC_ROOT / "ASVspoof5.eval.track_1.tsv"
PARQUET_DIR = REPO_ROOT / "data"
NUM_SHARDS = 200
EXPECTED_ROWS = 680774
EXPECTED_BONAFIDE = 138688
EXPECTED_SPOOF = 542086
TARGET_SR = 16000
PROBE_SAMPLE = 3000
WORKERS = int(os.environ.get("A5_BUILD_WORKERS", "64"))

FEATURES = Features(
    {
        "path": Value("string"),
        "audio": Audio(sampling_rate=16000),
        "label": ClassLabel(names=["bonafide", "spoof"]),
        "notes": Value("string"),
    }
)


def parse_metadata():
    """Parse track_1.tsv (10 whitespace-separated columns)."""
    records = []
    with open(META_PATH, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 10:
                raise ValueError(f"Expected 10 columns, got {len(parts)}: {line!r}")
            records.append(
                {
                    "speaker": parts[0],
                    "uid": parts[1],
                    "gender": parts[2],
                    "codec": parts[3],
                    "codec_id": parts[4],
                    "source_id": parts[5],
                    "attack_condition": parts[6],
                    "attack": parts[7],
                    "label": parts[8],
                }
            )
    return records


def build_notes(rec):
    return json.dumps(
        {
            "utterance_id": rec["uid"],
            "speaker_id": rec["speaker"],
            "gender": rec["gender"],
            "codec": rec["codec"],
            "codec_id": rec["codec_id"],
            "source_id": rec["source_id"],
            "attack_condition": rec["attack_condition"],
            "attack_id": rec["attack"],
        }
    )


def _probe_one(uid):
    try:
        data, _ = sf.read(str(FLAC_DIR / f"{uid}.flac"))
        if data.shape[0] == 0:
            return f"{uid}: empty"
        return None
    except Exception as e:  # noqa: BLE001
        return f"{uid}: {str(e).splitlines()[0][:100]}"


def probe_decodability(records):
    rng = random.Random(0)
    sample = rng.sample(records, min(PROBE_SAMPLE, len(records)))
    uids = [r["uid"] for r in sample]
    failures = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for err in ex.map(_probe_one, uids, chunksize=32):
            if err:
                failures.append(err)
    print(f"Probe: {len(failures)}/{len(uids)} sample clips failed soundfile decode")
    if failures:
        for f in failures[:10]:
            print(f"  {f}")
        raise RuntimeError(
            "Source FLAC no longer cleanly decodable; the CLEAN raw-embed path is "
            "unsafe. Re-introduce a re-encode stage (see ASVspoof2021_LA builder)."
        )


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


def _build_shard(task):
    """Worker: build one shard parquet from its row slice. Resumable + atomic."""
    shard_index, rows, num_shards = task
    shard_name = f"test-{shard_index:05d}-of-{num_shards:05d}.parquet"
    final = PARQUET_DIR / shard_name
    if final.exists() and final.stat().st_size > 0:
        return (shard_index, len(rows), "skipped")

    def row_gen():
        for rec in rows:
            uid = rec["uid"]
            yield {
                "path": f"{uid}.flac",
                "audio": {
                    "bytes": (FLAC_DIR / f"{uid}.flac").read_bytes(),
                    "path": f"{uid}.flac",
                },
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
    probe_decodability(records)

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
    tasks = [(i, rows, num_shards) for i, rows in enumerate(shards)]
    stage_workers = min(WORKERS, len(tasks))
    print(f"Building {len(tasks)} shard(s) with {stage_workers} workers...")
    built = skipped = 0
    with ProcessPoolExecutor(max_workers=stage_workers) as ex:
        for idx, n, status in tqdm(
            ex.map(_build_shard, tasks), total=len(tasks), desc="shards", unit="shard"
        ):
            if status == "built":
                built += 1
            elif status == "skipped":
                skipped += 1
    print(f"Done: {built} built, {skipped} skipped")

    _verify(num_shards, sample_mode)
    print("All verifications passed!")

    if not sample_mode:
        from speech_spoof_bench import labels

        out = labels.emit_labels(REPO_ROOT)
        print(f"Wrote {out}")


def _verify(num_shards, sample_mode):
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

- [ ] **Step 2: Byte-compile check (catches syntax errors fast)**

Run: `python3 -m py_compile /home/kirill/speech-spoof-bench/benchmarks/ASVspoof5/build_parquet.py && echo OK`
Expected: `OK`.

---

## Task 3: Write `eval.yaml` and `README.md`

**Files:**
- Modify (overwrite stub): `benchmarks/ASVspoof5/eval.yaml`
- Modify (overwrite stub): `benchmarks/ASVspoof5/README.md`

- [ ] **Step 1: Write `eval.yaml`**

Overwrite `benchmarks/ASVspoof5/eval.yaml`:

```yaml
name: ASVspoof5
description: >
  Track 1 (spoofing / deepfake detection) evaluation partition of ASVspoof 5.
  Binary classification: bonafide vs. spoof. EER computed on the official
  ASVspoof5 track_1 eval protocol (680,774 utterances).
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

- [ ] **Step 2: Write `README.md`**

Overwrite `benchmarks/ASVspoof5/README.md`:

```markdown
---
license: odc-by
language:
  - en
pretty_name: ASVspoof 5 (track 1, eval)
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
---

# ASVspoof 5 (track 1, eval)

Benchmark-ready packaging of the **Track 1 (spoofing / deepfake detection) evaluation partition** of the ASVspoof 5 challenge, for speech anti-spoofing and synthetic / deepfake voice detection.

## Overview

Track 1 is binary classification: **bonafide** (genuine human speech) vs. **spoof** (synthetic / converted speech). This packaging contains the full track_1 evaluation set. The original challenge is at https://www.asvspoof.org/.

## License & redistribution

Redistributed under the **Open Data Commons Attribution License (ODC-By) v1.0**. See `LICENSE.txt`. Labels and the evaluation protocol are unmodified; audio is the original 16 kHz mono FLAC, embedded bit-exactly (no re-encode — a full decode probe of all 680,774 clips passed cleanly).

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `path` | `string` | `<utterance_id>.flac`, unique |
| `audio` | `Audio(16000)` | 16 kHz mono FLAC |
| `label` | `ClassLabel` | `"bonafide"` (0) / `"spoof"` (1) |
| `notes` | `string` | JSON: `utterance_id`, `speaker_id`, `gender`, `codec`, `codec_id`, `source_id`, `attack_condition`, `attack_id` |

`notes` example:
```json
{"utterance_id": "E_0009538969", "speaker_id": "E_1607", "gender": "M", "codec": "C05", "codec_id": "2", "source_id": "E_0009486171", "attack_condition": "AC1", "attack_id": "A26"}
```

## Quick Start

```python
from datasets import load_dataset

ds = load_dataset("SpeechAntiSpoofingBenchmarks/ASVspoof5", split="test")
print(ds[0])
```

## Stats

| Stat | Value |
|------|-------|
| Total trials | 680,774 |
| Bonafide | 138,688 |
| Spoof | 542,086 |

## Source provenance

- Original challenge: https://www.asvspoof.org/
- Evaluation protocol: `ASVspoof5.eval.track_1.tsv`

## Evaluation

For evaluation instructions and submission format, see [`submissions/README.md`](submissions/README.md).

## Citation

```bibtex
@inproceedings{wang2024asvspoof5,
  title     = {{ASVspoof 5: Crowdsourced Speech Data, Deepfakes, and Adversarial Attacks at Scale}},
  author    = {Wang, Xin and Delgado, H{\'e}ctor and Tak, Hemlata and others},
  year      = {2024},
  booktitle = {ASVspoof Workshop 2024},
}
```

## Maintainer

Contact: k.n.borodin@mtuci.ru
```

- [ ] **Step 3: Fix the submissions template trial count**

The scaffold already substitutes the dataset name; only the trial count needs correcting. Edit `benchmarks/ASVspoof5/submissions/results_template.yaml`: confirm `dataset.id: SpeechAntiSpoofingBenchmarks/ASVspoof5` and set `scores.n_trials: 680774`.
Run: `grep -nE "id:|n_trials" /home/kirill/speech-spoof-bench/benchmarks/ASVspoof5/submissions/results_template.yaml`
Expected: id points at `SpeechAntiSpoofingBenchmarks/ASVspoof5`, `n_trials: 680774`.

---

## Task 4: Sample build + offline validate (fast gate)

**Files:**
- Output: `benchmarks/ASVspoof5/data/test-00000-of-00001.parquet`

- [ ] **Step 1: Build a 64-row sample**

Run:
```bash
cd /home/kirill/speech-spoof-bench/benchmarks/ASVspoof5
python3 build_parquet.py --limit 64
```
Expected: `Probe: 0/64 ... failed`, `SAMPLE MODE: 64 rows -> 1 shard`, `verify: 64 rows, row0 16000Hz ...s decodable OK`, `All verifications passed!`.

- [ ] **Step 2: Validate the local dir offline**

Run:
```bash
cd /home/kirill/speech-spoof-bench/benchmarks
speech-spoof-bench validate-dataset ./ASVspoof5 --skip-submissions
```
Expected: D1–D7 all green (PASS). If any check is red, fix the relevant file and re-run before proceeding.

- [ ] **Step 3: Remove the sample shard (the full build writes 200 shards)**

Run: `rm -f /home/kirill/speech-spoof-bench/benchmarks/ASVspoof5/data/test-00000-of-00001.parquet`
Expected: sample shard gone (so its `*-of-00001` name can't collide with the `*-of-00200` full build).

---

## Task 5: Full build (background) + verify + labels

**Files:**
- Output: `benchmarks/ASVspoof5/data/test-*.parquet` (200 shards) + `data/labels.parquet`

- [ ] **Step 1: Launch the full build in the background**

Run (background; 64 workers, sorted order):
```bash
cd /home/kirill/speech-spoof-bench/benchmarks/ASVspoof5
A5_BUILD_WORKERS=64 nohup python3 build_parquet.py > build.log 2>&1 &
echo "pid $!"
```
Expected: a pid; the build is resumable (re-running skips completed shards).

- [ ] **Step 2: Wait for completion, then check the log tail**

Run: `tail -20 /home/kirill/speech-spoof-bench/benchmarks/ASVspoof5/build.log`
Expected (on success): `Done: 200 built, 0 skipped`, `verify: 680774 rows, row0 16000Hz ...s decodable OK`, `All verifications passed!`, `Wrote .../data/labels.parquet`. The in-script asserts enforce 680774 / 138688 / 542086 — a count mismatch aborts with an AssertionError.

- [ ] **Step 3: Confirm shard count and size**

Run:
```bash
ls /home/kirill/speech-spoof-bench/benchmarks/ASVspoof5/data/test-*.parquet | wc -l
du -sh /home/kirill/speech-spoof-bench/benchmarks/ASVspoof5/data
ls -la /home/kirill/speech-spoof-bench/benchmarks/ASVspoof5/data/labels.parquet
```
Expected: 200 shards; ~80 GB total; `labels.parquet` present.

---

## Task 6: Final offline validation on the full build

- [ ] **Step 1: Validate the full local dataset offline**

Run:
```bash
cd /home/kirill/speech-spoof-bench/benchmarks
speech-spoof-bench validate-dataset ./ASVspoof5 --skip-submissions
```
Expected: D1–D7 all green against the full 200-shard build. Stop and fix anything red before pushing.

---

## Task 7: Push to HF + online validation

**Files:** none local; creates `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof5`.

- [ ] **Step 1: Create the repo and upload (resumable, LFS auto for parquet)**

Run:
```bash
python3 - <<'PY'
from huggingface_hub import HfApi, create_repo
repo = "SpeechAntiSpoofingBenchmarks/ASVspoof5"
create_repo(repo, repo_type="dataset", exist_ok=True)
HfApi().upload_large_folder(
    repo_id=repo,
    repo_type="dataset",
    folder_path="/home/kirill/speech-spoof-bench/benchmarks/ASVspoof5",
    ignore_patterns=["build.log", "__pycache__/*", "*.pyc", "_clean_flac/*"],
)
print("upload complete")
PY
```
Expected: `upload complete` (this transfers ~80 GB; `upload_large_folder` is resumable — re-run if interrupted).

- [ ] **Step 2: Validate the published dataset online**

Run:
```bash
speech-spoof-bench validate-dataset SpeechAntiSpoofingBenchmarks/ASVspoof5 --skip-submissions
```
Expected: D1–D7 all green against what HF serves.

- [ ] **Step 3: Capture the pinned commit SHA**

Run:
```bash
python3 - <<'PY'
from huggingface_hub import HfApi
info = HfApi().dataset_info("SpeechAntiSpoofingBenchmarks/ASVspoof5")
print("REVISION:", info.sha)
PY
```
Expected: a 40-char lowercase hex SHA. Record it as `<ASVSPOOF5_SHA>` for Task 8.

---

## Task 8: Manifest PR — add ASVspoof5 to Core

**Files:**
- Modify: `arena-manifest/manifest.yaml` (local clone) — append to `core_set`
- Modify: `arena-manifest/CHANGELOG.yaml` — append a `dataset_added` event
- Push: a PR on `SpeechAntiSpoofingBenchmarks/arena-manifest`

- [ ] **Step 1: Edit `manifest.yaml` — add to `core_set`**

In `/home/kirill/speech-spoof-bench/arena-manifest/manifest.yaml`, append under `core_set:` (keep alphabetical-ish ordering consistent with siblings):
```yaml
  - id: SpeechAntiSpoofingBenchmarks/ASVspoof5
    revision: <ASVSPOOF5_SHA>
```
Verify: `grep -nA1 "ASVspoof5" /home/kirill/speech-spoof-bench/arena-manifest/manifest.yaml`
Expected: the new id + a lowercase-hex revision (7–40 chars).

- [ ] **Step 2: Edit `CHANGELOG.yaml` — add the event**

Append under `events:` in `/home/kirill/speech-spoof-bench/arena-manifest/CHANGELOG.yaml`:
```yaml
  - {date: 2026-06-06, type: dataset_added, text: "ASVspoof5 added to Core (track_1 eval; 680,774 trials: 138,688 bonafide / 542,086 spoof)", dataset_id: "SpeechAntiSpoofingBenchmarks/ASVspoof5"}
```

- [ ] **Step 3: Validate the manifest locally (schema)**

Run:
```bash
cd /home/kirill/speech-spoof-bench/arena-manifest
speech-spoof-bench manifest 2>/dev/null >/dev/null && echo "manifest CLI ok"
python3 -c "import yaml; yaml.safe_load(open('manifest.yaml')); yaml.safe_load(open('CHANGELOG.yaml')); print('YAML parses')"
```
Expected: `YAML parses` (and no schema error). If the package exposes a manifest-schema validator, run it; otherwise the verify-hf workflow will schema-check on PR.

- [ ] **Step 4: Open the PR on arena-manifest**

The pinned SHA is already written into `manifest.yaml` (Task 8 Step 1). This uploads both edited files as a PR:
```bash
python3 - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
ops = []
from huggingface_hub import CommitOperationAdd
base = "/home/kirill/speech-spoof-bench/arena-manifest"
for f in ("manifest.yaml", "CHANGELOG.yaml"):
    ops.append(CommitOperationAdd(path_in_repo=f, path_or_fileobj=f"{base}/{f}"))
res = api.create_commit(
    repo_id="SpeechAntiSpoofingBenchmarks/arena-manifest",
    repo_type="dataset",
    operations=ops,
    commit_message="Add ASVspoof5 to Core (680,774 track_1 eval trials)",
    create_pr=True,
)
print("PR:", res.pr_url)
PY
```
Expected: a `PR: https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/arena-manifest/discussions/N` URL.

- [ ] **Step 5: Report the PR URL + note the Core-coverage impact**

Surface to the user: the PR URL, and that adding to **Core** re-computes every existing submission's coverage (deliberate). Maintainer merges after review; the Arena re-ingest then picks up ASVspoof5.

---

## Post-completion

Once the manifest PR is merged and ASVspoof5 appears on the Arena, proceed to the **skill** follow-up (separate task): wrap this scaffold→fork-builder→probe→benchmark-params→build→validate→push→manifest flow into a reusable skill for future ASVspoof-style datasets.

## Cleanup (optional)

- `benchmarks/_asvspoof5_probe.py` and `benchmarks/_asvspoof5_probe_report.json` are the throwaway decodability probe + its report. Keep the report as evidence or delete both after the dataset is live.
- `benchmarks/ASVspoof5/build.log` is gitignored noise; safe to delete after a successful build.
