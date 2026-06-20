# Runbook — clearing "missing reproduction block" warnings

The Arena's **About** tab can show a number of
`missing reproduction block — submission is unverified, skipped` warnings. This runbook
explains what they are and the exact steps to clear them.

## What the warning means

When the Arena ingests submissions, any `submissions/<slug>.yaml` whose `reproduction:`
block is empty is treated as **unverified**, so its row is **skipped** and a warning is
recorded (`arena/ingest.py`). These are almost always **superseded blockless
duplicates**: a second submission file for a (model, dataset) pair that *already* has a
valid, verified row. They are cosmetic cleanup, not a real coverage gap — clearing them
changes **no tier and no rank**.

## The one constraint that makes this tricky

**Verification only runs on submission files that are ADDED in a PR; an in-place edit is
ignored** (`verify-pr` compares the PR's files against `main` and acts only on the new
ones). So you **cannot** fix a blockless file by editing it in place — verification will
never look at it. You must **delete it, or delete-then-add a replacement.**

## Procedure

1. **List the offenders.** Read the warnings on the live About tab, or pull them from the
   Space's `cache.json` `warnings` array, filtering for `reason` containing
   `missing reproduction block`. Note each `submissions/<slug>.yaml` and its dataset repo.

2. **For each one, decide: delete or refill.**
   - **It's a duplicate** (a valid row for the same model+dataset already exists) →
     **delete** the blockless file. Open a PR on that dataset repo that *removes*
     `submissions/<slug>.yaml`. A delete needs no verification — it only removes.
   - **It's the only row** for that model+dataset → **refill** it. Because in-place edits
     are ignored, do a **delete-then-add**: remove the blockless file and add a new file
     (same slug) that **has** a populated `reproduction:` block (you reproduce the result
     yourself: `match: scoring`, with `reproduced_at`). The *add* triggers verification; a
     maintainer merges it.

3. **Re-ingest.** After the PRs merge, force a refresh (the Space's **🔄 Refresh** button,
   or the post-merge webhook) so the Arena re-reads submissions. The warning count drops.

4. **Record it.** Add a `note` line to `arena-manifest/CHANGELOG.yaml`, e.g.:
   ```yaml
   - {date: 2026-06-20, type: note, text: "Cleared blockless duplicate <slug>/<dataset>"}
   ```

## Done when

The About tab shows **0** "missing reproduction block" warnings, the cleanup is noted in
the `CHANGELOG.yaml`, and no tier or rank has changed (these were duplicates, by
definition).
