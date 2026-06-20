# In-the-Wild Dataset + random-baseline End-to-End Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish `SpeechAntiSpoofingBenchmarks/InTheWild` as a Core Arena dataset, score the `random-baseline` system against it, get it merged + verified, and badge it on the live Arena.

**Architecture:** Two sequenced sub-projects. **A** builds + publishes the dataset and opens the manifest PR; **B** scores random-baseline against the published dataset and badges it. B depends on A's published commit SHA, so A lands first. Claude runs all build/validate/push/run/submit/reproduce commands (`HF_TOKEN` present); the user only merges the two PRs (M1, M2) and triggers Arena re-ingest.

**Tech Stack:** `speech-spoof-bench` CLI, HuggingFace `datasets` + `hf` CLI, `soundfile`, `pyarrow`. Source: `/home/kirill/mnt/users_4tb/datasets/release_in_the_wild` (31,779 wavs, 16 kHz mono FLOAT, `meta.csv` = `file,speaker,label`, Apache-2.0).

**Spec:** `docs/specs/2026-05-30-inthewild-dataset-design.md`

---

## File Structure

Dataset repo (authored locally in `benchmarks/InTheWild/`, published via `hf upload`):

| File | Responsibility |
|---|---|
| `benchmarks/InTheWild/build_parquet.py` | Raw wavs + meta.csv → canonical 4-col parquet (8 shards) + verify |
| `benchmarks/InTheWild/README.md` | HF dataset card; D6 front-matter incl. `arena-ready` tag |
| `benchmarks/InTheWild/eval.yaml` | task block; `metrics: [eer_percent]` |
| `benchmarks/InTheWild/LICENSE.txt` | Apache-2.0 text + VocalSynthesis attribution |
| `benchmarks/InTheWild/.gitattributes` | LFS rule for `*.parquet` |
| `benchmarks/InTheWild/submissions/README.md` | submitter instructions |
| `benchmarks/InTheWild/submissions/results_template.yaml` | schema-v4 starting point |
| `benchmarks/InTheWild/data/test-*.parquet` + `data/labels.parquet` | generated |

Other repos:

| File | Responsibility |
|---|---|
| `arena-manifest/manifest.yaml` | add InTheWild to `core_set` at pinned SHA |
| `arena-manifest/CHANGELOG.yaml` | `dataset_added` + re-ingest `note` events |
| `benchmarks/InTheWild/submissions/random-baseline.yaml` | the submission PR (schema v4) |
| random-baseline-asas model card `README.md` (on HF) | dynamic tier+rank badges |

---

## Phase A — Dataset

### Task A1: Author the dataset repo metadata files

**Files:**
- Create: `benchmarks/InTheWild/README.md`
- Create: `benchmarks/InTheWild/eval.yaml`
- Create: `benchmarks/InTheWild/.gitattributes`
- Create: `benchmarks/InTheWild/submissions/README.md`
- Create: `benchmarks/InTheWild/submissions/results_template.yaml`
- Create: `benchmarks/InTheWild/LICENSE.txt`

- [ ] **Step 1: Create the repo dirs**

```bash
mkdir -p /home/kirill/speech-spoof-bench/benchmarks/InTheWild/submissions
mkdir -p /home/kirill/speech-spoof-bench/benchmarks/InTheWild/data
cd /home/kirill/speech-spoof-bench/benchmarks/InTheWild
```

- [ ] **Step 2: Write `README.md`**

```markdown
---
license: apache-2.0
language: [en]
pretty_name: In-the-Wild Audio Deepfake Dataset
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
paperswithcode_id:
arxiv:
  - "2203.16263"
---

# In-the-Wild Audio Deepfake Dataset

Benchmark-ready packaging of the **In-the-Wild** audio deepfake dataset for speech
anti-spoofing / synthetic-voice detection.

## Overview

In-the-Wild (Müller et al., *Does Audio Deepfake Detection Generalize?*, arXiv
2203.16263) pairs genuine speech with audio deepfakes of politicians and public
figures, collected from publicly available sources. It is a **cross-domain
generalization** benchmark: models trained on lab datasets (e.g. ASVspoof) are
evaluated here against real-world conditions. The task is binary classification:
**bonafide** (genuine human speech) vs. **spoof** (deepfake). 31,779 clips
(19,963 bonafide / 11,816 spoof), 16 kHz mono.

## License & redistribution

Redistributed under the **Apache License 2.0**; the full text is in `LICENSE.txt`.
Audio is the original 16 kHz mono signal encoded to FLAC (16-bit PCM). We thank
'VocalSynthesis' for the audio deepfakes included in the source dataset.

## Schema

Canonical 4-column parquet: `path` (string), `audio` (`Audio(16000)`), `label`
(`ClassLabel[bonafide, spoof]`), `notes` (JSON string with a unique
`utterance_id`, the `speaker` name, and the source `label` string).

## Citation

```bibtex
@inproceedings{muller2022does,
  title={Does Audio Deepfake Detection Generalize?},
  author={M{\"u}ller, Nicolas M and Czempin, Pavel and Dieckmann, Franziska and Froghyar, Adam and B{\"o}ttinger, Konstantin},
  booktitle={Interspeech},
  year={2022}
}
```
```

- [ ] **Step 3: Write `eval.yaml`**

```yaml
name: InTheWild
description: >
  In-the-Wild audio deepfake dataset (arXiv 2203.16263). Genuine speech and audio
  deepfakes of politicians and public figures collected from publicly available
  sources. 31,779 clips (19,963 bonafide / 11,816 spoof), 16 kHz mono. Binary
  classification: bonafide vs. spoof, scored by EER. A cross-domain
  generalization benchmark.
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

- [ ] **Step 4: Write `.gitattributes`**

```
*.parquet filter=lfs diff=lfs merge=lfs -text
```

- [ ] **Step 5: Write `submissions/results_template.yaml`**

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
  id: SpeechAntiSpoofingBenchmarks/InTheWild
  revision: ""
  split: test

scores:
  eer_percent: 0.0
  n_trials: 31779
  n_skipped: 0

artifact:
  # Must be pinned by commit sha. Pattern:
  #   https://huggingface.co/<owner>/<repo>/resolve/<commit-sha>/.eval_results/SpeechAntiSpoofingBenchmarks/InTheWild/scores.txt
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

- [ ] **Step 6: Write `submissions/README.md`**

Copy `benchmarks/CD-ADD/submissions/README.md` and replace every `ASVspoof2019_LA` with `InTheWild` and the `n_trials` example `71237` with `31779`:

```bash
sed -e 's/ASVspoof2019_LA/InTheWild/g' -e 's/71237/31779/g' \
  /home/kirill/speech-spoof-bench/benchmarks/CD-ADD/submissions/README.md \
  > /home/kirill/speech-spoof-bench/benchmarks/InTheWild/submissions/README.md
```

- [ ] **Step 7: Write `LICENSE.txt` (attribution header + Apache-2.0 text)**

```bash
cd /home/kirill/speech-spoof-bench/benchmarks/InTheWild
{
  echo "In-the-Wild Audio Deepfake Dataset"
  echo "Source: Müller et al., \"Does Audio Deepfake Detection Generalize?\" (arXiv 2203.16263)."
  echo "We thank 'VocalSynthesis' for the audio deepfakes included in the source dataset."
  echo
  echo "Redistributed under the Apache License, Version 2.0, reproduced below."
  echo "----------------------------------------------------------------------"
  echo
  curl -fsSL https://www.apache.org/licenses/LICENSE-2.0.txt
} > LICENSE.txt
test -s LICENSE.txt && grep -q "Apache License" LICENSE.txt && echo "LICENSE.txt OK"
```

Expected: `LICENSE.txt OK`. (If offline, paste the canonical Apache-2.0 text from `https://www.apache.org/licenses/LICENSE-2.0.txt` after the header instead.)

---

### Task A2: Write `build_parquet.py`

**Files:**
- Create: `benchmarks/InTheWild/build_parquet.py`

- [ ] **Step 1: Write the builder**

```python
"""Build the SpeechAntiSpoofingBenchmarks/InTheWild HF dataset repo.

In-the-Wild (arXiv 2203.16263, Apache-2.0) is a real-world audio-deepfake set:
genuine speech and deepfakes of politicians / public figures collected from
publicly available sources. 31,779 clips, 16 kHz mono.

Reads meta.csv (file,speaker,label), emits the canonical 4-column parquet
(path / audio / label / notes), writes NUM_SHARDS shards into data/.

Source wavs are 16 kHz mono FLOAT. Audio is embedded as FLAC PCM_16: read float
and let soundfile scale to 16-bit PCM on the FLAC *write* (reading a FLOAT wav
with dtype="int16" yields all zeros — libsndfile does not scale float->int on
read). This is a float->int16 conversion (NOT bit-exact), so verify() asserts
schema / row-0 / decodability / non-silence only. No resample (already 16 kHz).

Full build (default): every decodable clip into NUM_SHARDS shards.
Sample build: --limit N (or ITW_BUILD_LIMIT=N) -> first N rows into 1 shard,
skipping the full-count asserts (keeps schema / uniqueness / row-0 checks).
"""

import argparse
import csv
import io
import json
import os
import tempfile
from pathlib import Path

import soundfile as sf
from datasets import Audio, ClassLabel, Dataset, Features, Value

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = Path("/home/kirill/mnt/users_4tb/datasets/release_in_the_wild")
META_PATH = SRC_ROOT / "meta.csv"
PARQUET_DIR = REPO_ROOT / "data"
NUM_SHARDS = 8
TARGET_SR = 16000
EXPECTED_ROWS = 31779
EXPECTED_BONAFIDE = 19963
EXPECTED_SPOOF = 11816

LABEL_MAP = {"bona-fide": "bonafide", "spoof": "spoof"}

FEATURES = Features(
    {
        "path": Value("string"),
        "audio": Audio(sampling_rate=16000),
        "label": ClassLabel(names=["bonafide", "spoof"]),
        "notes": Value("string"),
    }
)


def catalogue() -> list[dict]:
    """One record per row of meta.csv (file,speaker,label)."""
    records: list[dict] = []
    with open(META_PATH, newline="") as f:
        for row in csv.DictReader(f):
            fname = row["file"].strip()
            raw_label = row["label"].strip()
            speaker = row["speaker"].strip()
            label = LABEL_MAP[raw_label]
            uid = f"ITW_{Path(fname).stem}"
            note = {"utterance_id": uid, "speaker": speaker, "label": raw_label}
            records.append(
                {
                    "abs": SRC_ROOT / fname,
                    "path": fname,
                    "label": label,
                    "notes": json.dumps(note),
                    "utterance_id": uid,
                }
            )
    return records


def flac_bytes(wav: Path) -> bytes:
    """Read a 16 kHz mono FLOAT wav and re-encode to FLAC PCM_16 in memory.
    Read float (default dtype) — reading a FLOAT wav as int16 yields all zeros —
    and let soundfile scale float [-1,1] to 16-bit PCM on the FLAC write (FLAC's
    default subtype is PCM_16). NOT bit-exact vs the float source (deliberate)."""
    data, sr = sf.read(str(wav))
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="FLAC")
    return buf.getvalue()


def probe_duration(wav: Path):
    """Duration in seconds if the clip fully decodes via soundfile, else None."""
    try:
        info = sf.info(str(wav))
        sf.read(str(wav))  # full decode catches corrupt bodies
        return info.frames / info.samplerate
    except Exception:  # noqa: BLE001 — reported by caller
        return None


def build() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    limit = args.limit
    if limit is None and os.environ.get("ITW_BUILD_LIMIT"):
        limit = int(os.environ["ITW_BUILD_LIMIT"])
    sample_mode = limit is not None

    print(f"Reading catalogue from {META_PATH} ...")
    records = catalogue()
    print(f"  {len(records)} rows in meta.csv")

    print("Probing decodability (full soundfile decode per clip)...")
    kept: list[dict] = []
    dropped: list[tuple[str, str]] = []
    for i, rec in enumerate(records, 1):
        dur = probe_duration(rec["abs"])
        if dur is None:
            dropped.append((rec["path"], "soundfile failed to decode"))
            continue
        rec["duration"] = dur
        kept.append(rec)
        if i % 4000 == 0:
            print(f"  probed {i}/{len(records)} (dropped so far: {len(dropped)})")
    print(f"Decodable: {len(kept)}; dropped (undecodable): {len(dropped)}")
    for p, err in dropped[:20]:
        print(f"  DROP {p}: {err}")

    # Deterministic order by numeric stem; guarantee row 0 >= 1.0 s for D3.
    kept.sort(key=lambda r: int(Path(r["path"]).stem))
    for i, rec in enumerate(kept):
        if rec["duration"] >= 1.0:
            if i != 0:
                kept[0], kept[i] = kept[i], kept[0]
            break
    else:
        raise RuntimeError("No clip >= 1.0 s found for row 0")

    if sample_mode:
        kept = kept[:limit]
        num_shards = 1
        print(f"SAMPLE MODE: first {len(kept)} rows into 1 shard")
    else:
        num_shards = NUM_SHARDS
        assert len(kept) == EXPECTED_ROWS, f"{len(kept)} != {EXPECTED_ROWS}"
        bonafide = sum(1 for r in kept if r["label"] == "bonafide")
        spoof = sum(1 for r in kept if r["label"] == "spoof")
        assert bonafide == EXPECTED_BONAFIDE, bonafide
        assert spoof == EXPECTED_SPOOF, spoof
        print(f"FULL BUILD: {len(kept)} rows "
              f"({bonafide} bonafide / {spoof} spoof) into {num_shards} shards")

    def row_gen():
        total = len(kept)
        for done, rec in enumerate(kept, 1):
            yield {
                "path": rec["path"],
                "audio": {"bytes": flac_bytes(rec["abs"]), "path": rec["path"]},
                "label": rec["label"],
                "notes": rec["notes"],
            }
            if done % 5000 == 0:
                print(f"  assembled {done}/{total}")

    print("Building dataset with Dataset.from_generator...")
    ds = Dataset.from_generator(row_gen, features=FEATURES)
    print(f"Dataset: {len(ds)} rows, features: {ds.features}")

    print(f"Writing {num_shards} shard(s) to {PARQUET_DIR} ...")
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(num_shards):
            shard = ds.shard(num_shards=num_shards, index=i)
            name = f"test-{i:05d}-of-{num_shards:05d}.parquet"
            shard.to_parquet(str(Path(tmpdir) / name))
            print(f"  wrote {name} ({len(shard)} rows)")
        for i in range(num_shards):
            name = f"test-{i:05d}-of-{num_shards:05d}.parquet"
            (PARQUET_DIR / name).write_bytes((Path(tmpdir) / name).read_bytes())

    verify(num_shards)
    print("All verifications passed!")


def verify(num_shards: int) -> None:
    import pyarrow.parquet as pq

    shards = sorted(PARQUET_DIR.glob("test-*.parquet"))
    total = sum(pq.read_metadata(str(f)).num_rows for f in shards)
    print(f"Verifying {len(shards)} shard(s), {total} rows...")

    uids, paths = set(), set()
    for f in shards:
        t = pq.read_table(str(f), columns=["path", "notes"])
        for p, n in zip(t.column("path").to_pylist(), t.column("notes").to_pylist()):
            paths.add(p)
            uids.add(json.loads(n)["utterance_id"])
    assert len(uids) == total, "duplicate utterance_id"
    assert len(paths) == total, "duplicate path"

    t0 = pq.read_table(str(shards[0]))
    assert set(t0.column_names) == {"path", "audio", "label", "notes"}, t0.column_names
    row0 = t0.column("audio")[0].as_py()
    assert isinstance(row0, dict) and row0.get("bytes"), "row 0 audio bytes missing"
    data, sr = sf.read(io.BytesIO(row0["bytes"]))
    dur = len(data) / sr
    assert sr == TARGET_SR, f"row 0 sr {sr} != {TARGET_SR}"
    assert dur >= 1.0, f"row 0 duration {dur:.2f}s < 1.0s"
    # Non-silence guard: catches a broken float->PCM encode that would store
    # all-zero (silent) audio yet still pass the sr/duration/schema checks.
    nonzero = int((data != 0).sum())
    assert nonzero > 0, "row 0 audio is all-zero (silent encode bug)"
    print(f"  row 0: {sr} Hz, {dur:.2f}s, {nonzero} nonzero samples, schema OK")


if __name__ == "__main__":
    build()
```

---

### Task A3: Sample build + offline validate (the fast gate)

**Files:** none created (generates a throwaway single shard).

- [ ] **Step 1: Sample build (50 rows, 1 shard)**

Run:
```bash
cd /home/kirill/speech-spoof-bench/benchmarks/InTheWild
ITW_BUILD_LIMIT=50 python build_parquet.py
```
Expected: prints `SAMPLE MODE: first 50 rows into 1 shard`, then `row 0: 16000 Hz, …s, schema OK` and `All verifications passed!`.

- [ ] **Step 2: Validate offline**

Run:
```bash
cd /home/kirill/speech-spoof-bench
speech-spoof-bench validate-dataset ./benchmarks/InTheWild --skip-submissions
```
Expected: D1–D7 all PASS (report `ok`). If any D-check fails, fix `build_parquet.py` / metadata and re-run Steps 1–2 before proceeding.

- [ ] **Step 3: Remove the sample shard**

Run:
```bash
rm -f /home/kirill/speech-spoof-bench/benchmarks/InTheWild/data/test-00000-of-00001.parquet
```

---

### Task A4: Full build + labels + offline validate

**Files:**
- Create: `benchmarks/InTheWild/data/test-0000{0..7}-of-00008.parquet`
- Create: `benchmarks/InTheWild/data/labels.parquet`

- [ ] **Step 1: Full build (all 31,779 rows, 8 shards)**

Run:
```bash
cd /home/kirill/speech-spoof-bench/benchmarks/InTheWild
python build_parquet.py
```
Expected: `FULL BUILD: 31779 rows (19963 bonafide / 11816 spoof) into 8 shards`, 8 `wrote test-…` lines, `All verifications passed!`. If `dropped (undecodable)` > 0, the asserts will fail (count mismatch) — investigate the dropped clips before continuing.

- [ ] **Step 2: Emit `data/labels.parquet`**

Run:
```bash
cd /home/kirill/speech-spoof-bench
speech-spoof-bench emit-labels ./benchmarks/InTheWild
```
Expected: writes `data/labels.parquet` and reports it matches the shards.

- [ ] **Step 3: Validate the full local repo offline**

Run:
```bash
speech-spoof-bench validate-dataset ./benchmarks/InTheWild --skip-submissions
```
Expected: D1–D7 all PASS.

---

### Task A5: Push to HuggingFace + online validate + record SHA

**Files:** none locally; publishes the repo.

- [ ] **Step 1: Upload the whole repo folder (creates the dataset repo)**

Run:
```bash
cd /home/kirill/speech-spoof-bench/benchmarks/InTheWild
hf upload SpeechAntiSpoofingBenchmarks/InTheWild . . \
  --repo-type dataset \
  --commit-message "Add In-the-Wild dataset (31,779 clips; 19,963 bonafide / 11,816 spoof)"
```
Expected: uploads README/eval.yaml/LICENSE/build_parquet.py/submissions/data shards; prints a commit URL. (`hf` handles LFS for the parquet automatically.)

- [ ] **Step 2: Record the commit SHA**

Run:
```bash
python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('SpeechAntiSpoofingBenchmarks/InTheWild').sha)"
```
Record the printed SHA as `<ITW_SHA>` — used in A6 and B2.

- [ ] **Step 3: Validate the published repo online**

Run:
```bash
cd /home/kirill/speech-spoof-bench
speech-spoof-bench validate-dataset SpeechAntiSpoofingBenchmarks/InTheWild --skip-submissions
```
Expected: D1–D7 all PASS against what HF serves.

---

### Task A6: Manifest PR (add to Core) → 🛠 M1 user merge

**Files:**
- Modify: `arena-manifest/manifest.yaml` (append to `core_set`)
- Modify: `arena-manifest/CHANGELOG.yaml` (append two `events`)

- [ ] **Step 1: Add InTheWild to `core_set` in `manifest.yaml`**

Append after the `CD-ADD` entry (use the real `<ITW_SHA>` from A5; must match `^[0-9a-f]{7,40}$`):
```yaml
  - id: SpeechAntiSpoofingBenchmarks/InTheWild
    revision: <ITW_SHA>
```

- [ ] **Step 2: Add changelog events in `CHANGELOG.yaml`**

Append under `events:`:
```yaml
  - {date: 2026-05-30, type: dataset_added, text: "InTheWild added to Core (31,779 real-world clips: 19,963 bonafide / 11,816 spoof)", dataset_id: "SpeechAntiSpoofingBenchmarks/InTheWild"}
  - {date: 2026-05-30, type: note, text: "Re-ingest to subscribe InTheWild for webhook routing"}
```
(Data change only — do NOT bump `schema_version` / `ranking_version`.)

- [ ] **Step 3: Open the manifest PR**

Run:
```bash
cd /home/kirill/speech-spoof-bench/arena-manifest
hf upload SpeechAntiSpoofingBenchmarks/arena-manifest \
  manifest.yaml manifest.yaml --repo-type dataset --create-pr \
  --commit-message "Add InTheWild to Core"
hf upload SpeechAntiSpoofingBenchmarks/arena-manifest \
  CHANGELOG.yaml CHANGELOG.yaml --repo-type dataset --create-pr \
  --commit-message "Changelog: InTheWild added to Core"
```
Expected: prints PR URL(s). If both files must land in one PR, instead push them together on a single PR branch — note the PR URL either way.

- [ ] **Step 4: 🛠 STOP — hand off to user (M1)**

Tell the user: "Manifest PR open at <URL(s)>. Please merge it and trigger an Arena re-ingest to subscribe InTheWild for webhook routing. Reply when merged + re-ingested." Wait for confirmation before Phase B.

---

## Phase B — random-baseline end-to-end (after A lands)

### Task B1: Run random-baseline against InTheWild

**Files:**
- Create: `results/InTheWild/scores.txt`
- Create: `results/InTheWild/result.yaml`

- [ ] **Step 1: Run the baseline**

Run:
```bash
cd /home/kirill/speech-spoof-bench
speech-spoof-bench run \
  --model-module speech_spoof_bench.examples.random_baseline:RandomBaseline \
  --datasets SpeechAntiSpoofingBenchmarks/InTheWild \
  --output-dir ./results
```
Expected: writes `results/InTheWild/scores.txt` (one `<utt_id> <score>` per line, 31,779 lines) and `results/InTheWild/result.yaml` (with `eer_percent` ≈ 50, `n_trials: 31779`, `n_skipped: 0`, `bench_version`).

- [ ] **Step 2: Sanity-check the result**

Run:
```bash
cat results/InTheWild/result.yaml
wc -l results/InTheWild/scores.txt
```
Expected: `eer_percent` near 50; `n_skipped` ≈ 0; `scores.txt` has ~31,779 lines (`len + n_skipped == 31779`). If `n_skipped` > 0, note the count for B2.

---

### Task B2: Upload scores + author the submission YAML

**Files:**
- Create: `benchmarks/InTheWild/submissions/random-baseline.yaml`

- [ ] **Step 1: Upload `scores.txt` to the model repo (commit-pinned)**

Run:
```bash
cd /home/kirill/speech-spoof-bench
hf upload SpeechAntiSpoofingBenchmarks/random-baseline-asas \
  results/InTheWild/scores.txt \
  .eval_results/SpeechAntiSpoofingBenchmarks/InTheWild/scores.txt \
  --repo-type model \
  --commit-message "Add InTheWild scores"
python -c "from huggingface_hub import HfApi; print(HfApi().model_info('SpeechAntiSpoofingBenchmarks/random-baseline-asas').sha)"
```
Record the printed model SHA as `<MODEL_SHA>`.

- [ ] **Step 2: Compute the scores sha256**

Run:
```bash
sha256sum results/InTheWild/scores.txt | awk '{print $1}'
```
Record as `<SCORES_SHA256>`. Also read `bench_version` and `eer_percent` from `results/InTheWild/result.yaml` (call them `<BENCH_VERSION>`, `<EER>`, and `<N_SKIPPED>`).

- [ ] **Step 3: Write `submissions/random-baseline.yaml`**

The `system` block is copied verbatim from the workspace-root `random-baseline-meta.yaml`. Fill the four runtime values (`<ITW_SHA>` from A5; `<EER>`, `<N_SKIPPED>`, `<MODEL_SHA>`, `<SCORES_SHA256>`, `<BENCH_VERSION>` from B1/B2):

```yaml
schema_version: 4

system:
  name: random-baseline
  slug: random-baseline
  description: >
    Reference random baseline. Returns N(0, 1) for every utterance using a
    fixed seed (seed=0). EER ≈ 50% by construction. Used as the seeded
    smoke-test baseline for the arena (roadmap Phase 3).
  code: https://github.com/SpeechAntiSpoofingBenchmarks/speech-spoof-bench
  checkpoint: https://huggingface.co/SpeechAntiSpoofingBenchmarks/random-baseline-asas
  params_millions: 1
  paper:
    arxiv_id: "1911.01601"
    url: https://arxiv.org/abs/1911.01601
    bibtex: |
      @article{wang2020asvspoof,
        title={ASVspoof 2019: A large-scale public database of synthesized,
               converted and replayed speech},
        author={Wang, Xin and Yamagishi, Junichi and Todisco, Massimiliano and
                Delgado, H{\'e}ctor and Nautsch, Andreas and Evans, Nicholas
                and Sahidullah, Md and Vestman, Ville and Kinnunen, Tomi and
                Lee, Kong Aik and others},
        journal={Computer Speech \& Language},
        volume={64},
        pages={101114},
        year={2020},
        publisher={Elsevier}
      }

dataset:
  id: SpeechAntiSpoofingBenchmarks/InTheWild
  revision: <ITW_SHA>
  split: test

scores:
  eer_percent: <EER>
  n_trials: 31779
  n_skipped: <N_SKIPPED>

artifact:
  scores_url: https://huggingface.co/SpeechAntiSpoofingBenchmarks/random-baseline-asas/resolve/<MODEL_SHA>/.eval_results/SpeechAntiSpoofingBenchmarks/InTheWild/scores.txt
  scores_sha256: <SCORES_SHA256>
  bench_version: "<BENCH_VERSION>"

reproduction: {}

submitter:
  hf_username: enderfry16
  contact: enderfry16@gmail.com

submitted_at: "2026-05-30"
notes: >
  Random baseline has no associated paper of its own. The paper field cites the
  ASVspoof 2019 dataset paper as a placeholder, consistent with the other
  random-baseline submissions (same system slug). The reproduction block is left
  empty for the maintainer to fill at merge.
```

- [ ] **Step 4: Validate the submission schema offline**

Run:
```bash
cd /home/kirill/speech-spoof-bench
speech-spoof-bench validate-submission ./benchmarks/InTheWild/submissions/random-baseline.yaml
```
Expected: schema PASS (no network).

---

### Task B3: Reproduce (the CI gate)

**Files:** none.

- [ ] **Step 1: Reproduce against the pinned revision**

Run:
```bash
cd /home/kirill/speech-spoof-bench
speech-spoof-bench reproduce ./benchmarks/InTheWild/submissions/random-baseline.yaml --scoring --no-local
```
Expected: downloads `scores.txt`, sha256 matches `<SCORES_SHA256>`, streams labels at `<ITW_SHA>`, recomputes EER and matches `<EER>` within 1e-6, coverage `len(scores) + n_skipped == 31779`. Exit 0. If it fails, fix the YAML/scores per the new-model.md failure table and re-run B2–B3.

---

### Task B4: Open the submission PR → 🛠 M2 user merge

**Files:** none locally; opens a PR on the dataset repo.

- [ ] **Step 1: Open the PR**

Run:
```bash
cd /home/kirill/speech-spoof-bench/benchmarks/InTheWild/submissions
hf upload SpeechAntiSpoofingBenchmarks/InTheWild \
  random-baseline.yaml submissions/random-baseline.yaml \
  --repo-type dataset --create-pr \
  --commit-message "Add random-baseline submission"
```
Expected: prints PR URL.

- [ ] **Step 2: 🛠 STOP — hand off to user (M2)**

Tell the user: "Submission PR open at <URL>. Please run the maintainer reproduce + fill the `reproduction:` block + merge (CI mirrors `reproduce --scoring --no-local`, already green locally). Reply when merged." Wait for confirmation.

---

### Task B5: Add dynamic badges to the model card

**Files:**
- Modify: `random-baseline-asas` model card `README.md` (fetched from HF, edited, re-uploaded)

- [ ] **Step 1: Fetch the current model card**

Run:
```bash
cd /home/kirill/speech-spoof-bench
hf download SpeechAntiSpoofingBenchmarks/random-baseline-asas README.md \
  --repo-type model --local-dir ./_modelcard
```
Expected: `./_modelcard/README.md` present.

- [ ] **Step 2: Insert the dynamic tier + rank badges**

Add these two lines near the top of `./_modelcard/README.md` (below the title; skip if an identical pair is already present from a prior dataset):
```markdown
[![arena tier](https://img.shields.io/endpoint?url=https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/random-baseline/tier.json)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=random-baseline)
[![arena rank](https://img.shields.io/endpoint?url=https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/random-baseline/rank.json)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=random-baseline)
```

- [ ] **Step 3: Upload the updated model card**

Run:
```bash
cd /home/kirill/speech-spoof-bench
hf upload SpeechAntiSpoofingBenchmarks/random-baseline-asas \
  ./_modelcard/README.md README.md \
  --repo-type model --commit-message "Add Arena tier/rank badges"
```
Expected: prints commit URL.

---

### Task B6: Verify on the live Arena

**Files:** none.

- [ ] **Step 1: Confirm the dynamic badges resolve**

Run:
```bash
curl -s "https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/random-baseline/tier.json"
curl -s "https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/random-baseline/rank.json"
```
Expected: JSON with a real `message` (a tier name / `#N of M`), not `unranked`. If `unranked`, the Arena hasn't re-ingested — ask the user to refresh (M1/M2 ingest).

- [ ] **Step 2: Confirm InTheWild coverage**

Open the Arena Space (`?system=random-baseline`) and confirm `random-baseline` shows an InTheWild EER ≈ 50%. Report the final state to the user.

---

## Self-Review notes

- **Spec coverage:** A1 (repo files) ↔ spec A1; A2 (builder) ↔ spec A2; A3–A4 (validate/build/labels) ↔ spec A3 steps 1–2; A5 (push/online-validate/SHA) ↔ spec A3 steps 3–5; A6 (manifest PR + M1) ↔ spec A3 steps 6–7; B1 ↔ spec B1; B2–B4 (upload/author/reproduce/PR + M2) ↔ spec B2–3; B5 (badges) ↔ spec B4; B6 (Arena verify) ↔ spec B5. No spec section unmapped.
- **Placeholders:** the only `<…>` tokens are runtime values captured by an explicit prior step (`<ITW_SHA>`, `<MODEL_SHA>`, `<SCORES_SHA256>`, `<BENCH_VERSION>`, `<EER>`, `<N_SKIPPED>`) — each is recorded in the step that produces it.
- **Type consistency:** `flac_bytes`, `probe_duration`, `catalogue`, `verify`, `build` signatures match between A2's source and its uses; `utterance_id` format `ITW_<stem>` is consistent across builder, notes, and reproduce coverage.
