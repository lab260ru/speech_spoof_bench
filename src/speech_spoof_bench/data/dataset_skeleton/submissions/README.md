# Benchmark Submissions

To submit a result, you'll upload two files (no git clone required):

1. **`scores.txt`** to **your own HF model repo** under `.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt`.
2. **`<your-slug>.yaml`** as a pull request to **this dataset repo** under `submissions/<your-slug>.yaml`.

The YAML in this repo carries a pinned URL pointing at your `scores.txt`, plus its sha256. Scores files do not live in this repo.

## Submitter workflow

### 1. Generate `scores.txt` locally

```bash
speech-spoof-bench run \
  --model-module <your_package>:<YourModelClass> \
  --datasets SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
```

Output: `results/ASVspoof2019_LA/scores.txt` (one line per utterance, `<utterance_id> <score>`, higher = more bonafide).

### 2. Upload `scores.txt` to your model repo

```bash
huggingface-cli upload <your-owner>/<your-model-repo> \
  results/ASVspoof2019_LA/scores.txt \
  .eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt \
  --repo-type=model \
  --commit-message="Add ASVspoof2019_LA scores"
```

**Note the commit sha** the CLI prints — you'll need it in the next step.

### 3. Fill in the submission YAML

Copy `results_template.yaml` to `<your-slug>.yaml` and fill in every field. The two most important fields:

- `artifact.scores_url`: the **pinned** URL to your uploaded scores file. Use the commit sha from step 2, not `main`:
  ```
  https://huggingface.co/<your-owner>/<your-model-repo>/resolve/<commit-sha-from-step-2>/.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt
  ```
  URLs with `/resolve/main/` are rejected because they're mutable.
- `artifact.scores_sha256`: `sha256sum results/ASVspoof2019_LA/scores.txt | awk '{print $1}'`.

Leave the `reproduction:` block empty — the maintainer fills it in at merge time.

### 4. Open the PR via HF CLI

```bash
huggingface-cli upload \
  SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA \
  <your-slug>.yaml submissions/<your-slug>.yaml \
  --repo-type=dataset \
  --create-pr \
  --commit-message="Add <your-slug> submission"
```

The CLI prints a PR URL. That's it.

### 5. Wait for maintainer reproduction

A maintainer runs `speech-spoof-bench reproduce --scoring <PR-branch>`, which:
- Fetches `scores_url`.
- Verifies the sha256 against `artifact.scores_sha256`.
- Recomputes EER from the file.
- Compares to your claimed `scores.eer_percent` (must match within 1e-6).

If it passes, the maintainer fills in `reproduction:` and merges. If it fails, you get a comment on the PR explaining why.

## Verification levels

| Level | What the maintainer checks | Cost |
|---|---|---|
| `scoring` (default) | sha + recomputed EER from your `scores.txt`. | Seconds. |
| `inference` (optional, follow-up) | Re-runs your checkpoint end-to-end and regenerates `scores.txt`. Must match within 0.05% EER. | Expensive. |

Submissions without a `reproduction:` block never appear in the arena.

## What about git clone + push?

You can do it that way too, but for a single 2 KB YAML it's massively heavier. The HF CLI path is the documented one.
