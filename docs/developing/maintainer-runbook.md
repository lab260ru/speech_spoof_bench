# Maintainer Runbook

Operational checklist for accepting model submissions, dataset additions, and
Arena manifest changes.

## Model Submission PR

Use this for PRs on a dataset repo under `submissions/*.yaml`.

1. Verify the PR branch:

   ```bash
   speech-spoof-bench ci verify-pr \
     --repo SpeechAntiSpoofingBenchmarks/<DATASET> \
     --pr <HF_DISCUSSION_NUMBER> \
     --branch refs/pr/<HF_DISCUSSION_NUMBER>
   ```

2. Require all three checks to be green: schema, `scores_sha256`, and metric match.
3. Merge only after the CI comment says `all checks passed`.
4. After merge, post the badge snippet:

   ```bash
   speech-spoof-bench ci post-merge-badge \
     --repo SpeechAntiSpoofingBenchmarks/<DATASET> \
     --pr <HF_DISCUSSION_NUMBER> \
     --sha <MERGE_COMMIT_SHA>
   ```

5. Confirm the Arena refreshes and the model appears in the expected tier/table.

## Dataset Addition

Use this before adding a dataset to `arena-manifest`.

1. Validate locally while the builder can still be fixed cheaply:

   ```bash
   speech-spoof-bench validate-dataset ./<dataset-dir>
   ```

2. Upload the dataset repo and validate the HF copy without using local registry state:

   ```bash
   speech-spoof-bench validate-dataset <owner>/<dataset> --no-local
   ```

3. Ensure `data/labels.parquet` exists. It keeps reproduce, nightly, and online
   validation metadata-only for labels:

   ```bash
   speech-spoof-bench emit-labels ./<dataset-dir>
   ```

4. Add at least one verified baseline submission before adding the dataset to the
   Arena manifest. The Arena skips submissions without a reproduction block.

## Arena Manifest Preview

Use this for PRs on `SpeechAntiSpoofingBenchmarks/arena-manifest`.

```bash
speech-spoof-bench ci preview-manifest \
  --repo SpeechAntiSpoofingBenchmarks/arena-manifest \
  --pr <HF_DISCUSSION_NUMBER> \
  --branch refs/pr/<HF_DISCUSSION_NUMBER>
```

Expected output:

- `Rows` increases when a dataset with verified submissions is added.
- `Warnings` is `0`.
- `Added datasets` and `Removed datasets` match the intended manifest delta.

The same check is available as the GitHub Actions workflow
`preview-arena-manifest` for manual dispatch.

## Live Arena Check

After merge/deploy, verify the public Space:

```bash
curl -fsS https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/healthz
curl -fsS https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/random-baseline/tier.json
```

Then check the UI tabs: Overview, Datasets, Per dataset, By model size, Over time,
Submit, and About.

## Cleanup Policy

Temporary validation repos are useful as evidence while a PR is open. After the PR
is merged or closed and the evidence is no longer needed, delete temporary model
and dataset repos or mark them clearly as fixtures. Do not merge temporary rows
into public dataset repos or `arena-manifest`.
