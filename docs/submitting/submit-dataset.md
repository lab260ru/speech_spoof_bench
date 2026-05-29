# Submit a dataset — step-by-step

This guide assumes no prior knowledge. By the end you'll have a new anti-spoofing
benchmark in the org that models can be evaluated on.

> 💡 Not sure your dataset qualifies, or need help hosting large audio? Email
> **[k.n.borodin@mtuci.ru](mailto:k.n.borodin@mtuci.ru)** before you start — it can save
> you a lot of work.

## What you're actually doing

A "dataset" here is a Hugging Face (HF) **dataset repo** containing the audio (as
parquet), a description, an evaluation config, and a folder where model results
accumulate. You:

1. confirm you're allowed to redistribute the audio,
2. generate the standard repo skeleton,
3. convert your audio into the canonical format,
4. fill in the description / eval config / citation,
5. validate locally until everything is green,
6. push to HF and open a PR adding it to the Arena's manifest.

## Prerequisites

- **Python 3.10+** and a **Hugging Face account** (https://huggingface.co/join).
- The HF CLI, logged in:
  ```bash
  pip install huggingface_hub && huggingface-cli login
  ```
- `pip install speech-spoof-bench`.
- Membership/permission to push under the `SpeechAntiSpoofingBenchmarks` org (ask the
  maintainer if you don't have it).

## Step 1 — Redistribution check (do this FIRST)

We only host datasets we can **legally redistribute under their original license**.

- ✅ OK: the license permits rehosting the audio (you'll ship the verbatim
  `LICENSE.txt`).
- ❌ Not OK: "download it yourself" loader-only repos, or anything you can't legally
  re-share.

If you're unsure, email us before building anything.

## Step 2 — Generate the repo skeleton

```bash
speech-spoof-bench scaffold-dataset \
  --name ASVspoof2021_DF --output-dir ./ASVspoof2021_DF
```

**Naming:** `<Source><Year>_<Partition>` for challenge sets (e.g. `ASVspoof2019_LA`);
a plain name for real-world sets (e.g. `InTheWild`). This creates the standard files
(README, `eval.yaml`, `submissions/`, a build script stub).

## Step 3 — Convert your audio to the canonical format

Build a parquet where **every row is exactly these four fields**:

| field | type | meaning |
|---|---|---|
| `path` | string | a stable, **unique** path/id for the clip |
| `audio` | Audio @ **16 kHz** | mono; resample during the build |
| `label` | ClassLabel `[bonafide, spoof]` | **index 0 = bonafide**, 1 = spoof |
| `notes` | string containing JSON | must parse and include a unique `utterance_id` |

Example `notes` value: `{"utterance_id": "LA_E_2834763", "speaker_id": "LA_0039"}`.
The scorer only needs `utterance_id`; everything else in `notes` is informational.

## Step 4 — Fill in README, eval.yaml, and citation

- **README front-matter (YAML at the top):** include the **`arena-ready`** tag — this is
  literally how the Arena discovers your dataset — plus the `arxiv:` id list, license,
  and `pretty_name`.
- **`eval.yaml`:** the task config and a `metrics:` list, e.g.:
  ```yaml
  metrics:
    - eer_percent
  ```
  The **first** metric is the dataset's *primary* metric — the one the leaderboard ranks
  by. `eer_percent` is the standard.
- **Citation section** in the README body: the paper's arXiv link and a BibTeX block.

## Step 5 — Validate until green

```bash
speech-spoof-bench validate-dataset ./ASVspoof2021_DF
```

This checks: the schema is exactly the four fields; labels are `[bonafide, spoof]`;
`notes` parses and every `utterance_id` is unique; the audio is 16 kHz; the README
front-matter has the required keys; and `eval.yaml` is well-formed with registered
metrics. **Fix every issue it reports** before moving on — the maintainer runs the same
check.

## Step 6 — Push, then open a manifest PR

1. Push your validated repo to
   `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/<name>`.
2. Open a pull request on the **`arena-manifest`** repo adding your dataset with a
   **pinned `revision`** (a commit sha — the exact version models will be scored
   against):
   ```yaml
   extended:
     - id: SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF
       revision: <commit-sha>
   ```

## Step 7 — Core vs Extended

- **Core** — counts toward tier coverage (Gold/Silver/Bronze) and the global rank.
- **Extended** — shown and rankable on its own tab, but doesn't affect tiers.

New datasets normally start in **Extended** and get promoted to **Core** once they've
proven stable and have a few submissions.

## Common mistakes

- **Audio not resampled to 16 kHz** — validation will reject it.
- **Duplicate or missing `utterance_id`** — must be present and unique in `notes`.
- **Forgot the `arena-ready` tag** — the Arena won't find the dataset without it.
- **Unpinned manifest revision** — you must pin a commit sha, not a branch name.

---

Questions at any step? Email **[k.n.borodin@mtuci.ru](mailto:k.n.borodin@mtuci.ru)**.
