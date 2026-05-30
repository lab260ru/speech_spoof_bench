# CI/CD: Webhook → GitHub Actions

There is **no inbound webhook on GitHub.** The trigger path is: Hugging Face → the
Arena's `/webhook` → the Arena dispatches a GitHub Action via the REST API
(`workflow_dispatch`). The Actions then call the package's `ci` CLI commands and comment
back on the HF discussion. This doc traces that whole loop.

## The HF webhook

Configured once, **org-wide**, at `https://huggingface.co/settings/webhooks`:

- **Watch:** the `SpeechAntiSpoofingBenchmarks` org.
- **Trigger:** *Repo update* only.
- **Target:** `https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/webhook`
- **Secret:** the value of `HF_WEBHOOK_SECRET` (HF sends it in the `X-Webhook-Secret`
  header; the handler validates before doing anything → 401 on mismatch).

## `webhook.py` — routing logic

`POST /webhook` (`hf_webhook`):

1. **Validate** the secret (`_check_secret` via `HF_WEBHOOK_SECRET`). Bad → 401.
2. **Filter** out events that aren't `scope == "repo.content"`, are from a `space` (the
   self-loop guard), or are for a repo not in the manifest (`_subscribed_repos()`, read
   from the in-memory manifest, falling back to `fetch_manifest()`).
3. **Deduplicate**: `_is_duplicate(repo, sha)` ignores the same `repo+sha` seen within a
   **300 s** window (HF retries), via an `OrderedDict` with monotonic-time eviction.
4. **Route** by ref:

| Event | Ref shape | Action |
|-------|-----------|--------|
| **PR update** | `refs/pr/N` | `_dispatch_verify_workflow(repo, N, ref)` → dispatch `verify-hf-pr.yml`. Response `scheduled/verify-pr`. |
| **Merge to main** | `refs/heads/main`, or `headSha` with no `updatedRefs` | Background `_refresh_and_commit(repo)` (re-ingest + commit cache). If a PR number is known, **also** `_dispatch_post_merge_workflow(repo, N, sha)` → dispatch `post-merge-badge.yml`. Response `scheduled/refresh[+post-merge]`. |

> **HF webhook v3 quirk.** v3 merge payloads don't carry the PR number. The handler
> recovers it from the merge commit's title suffix `(#N)` via
> `_pr_num_from_merge_commit()` (regex `\(#(\d+)\)\s*$`). If HF ever changes that suffix
> format, post-merge badges silently stop firing.

### The GitHub dispatch call

```python
requests.post(
    f"{_GH_API}/repos/{GH_VERIFY_WORKFLOW_REPO}/actions/workflows/verify-hf-pr.yml/dispatches",
    headers={"Authorization": f"Bearer {GH_PAT}", "Accept": "application/vnd.github+json"},
    json={"ref": "main", "inputs": {"repo": repo, "pr": str(pr), "branch": branch}},
    timeout=10,
)
```

`GH_VERIFY_WORKFLOW_REPO` defaults to `lab260ru/speech_spoof_bench`. The post-merge
dispatch is identical but targets `post-merge-badge.yml` with `inputs={repo, pr, sha}`,
and is **fire-and-log** (errors logged, don't fail the webhook).

## The four GitHub Actions workflows

All live in `.github/workflows/` of the package repo. Each does the same boilerplate:
checkout → setup Python → install the package → run a `speech-spoof-bench ci ...`
command. They differ only in trigger and command.

### `verify-hf-pr.yml` — verify a PR
```yaml
on:
  workflow_dispatch:
    inputs:
      repo:   { description: "HF dataset id (org/name)", required: true, type: string }
      pr:     { description: "HF PR / discussion number", required: true, type: string }
      branch: { description: "Branch ref on the dataset repo (e.g. refs/pr/42)", required: true, type: string }
```
Runs `speech-spoof-bench ci verify-pr --repo … --pr … --branch …`. Needs **`HF_BOT_TOKEN`**
(to comment) and `GH_RUN_URL` (composed from the GitHub context). → `verify_pr.run()`:
diff the branch against `main` to find **added** submissions, schema-check each, run
`reproduce --scoring` (tolerance `1e-6`), and post a Markdown verdict table (✅/❌/—) to
the HF discussion. Exit 0 if all pass, 1 otherwise.

### `nightly-revalidate.yml` — daily health check
```yaml
on:
  schedule:
    - cron: "0 6 * * *"   # 06:00 UTC daily
  workflow_dispatch:
```
`permissions: issues: write`. Runs `speech-spoof-bench ci nightly-revalidate --open-issues`.
Needs **`HF_BOT_TOKEN`** and **`GH_TOKEN`** (the built-in `GITHUB_TOKEN`, used by the `gh`
CLI). → `nightly.run()`: fetch the manifest, walk every merged submission across
core+extended, `reproduce --scoring` each, and manage GitHub issues labelled
**`stale-submission`** — open one per new failure (title `[<dataset_id>] <slug>`), comment
on persisting ones, close resolved ones. `run()` always returns 0 (so a detected drift
doesn't mark the Action red — the signal is the issue, not the job status).

### `post-merge-badge.yml` — badge a merge
```yaml
on:
  workflow_dispatch:
    inputs:
      repo: { description: "HF dataset id (org/name)", required: true, type: string }
      pr:   { description: "HF PR / discussion number", required: true, type: string }
      sha:  { description: "Merge commit sha on the dataset main branch", required: true, type: string }
```
Runs `speech-spoof-bench ci post-merge-badge --repo … --pr … --sha …`. Needs
**`HF_BOT_TOKEN`** + `GH_RUN_URL`. → `post_merge_badge.run()`: diff the merge commit
`<sha>` against its **parent** (not `main`, because the file is already on main) to find
added submissions, read the dataset's primary metric from `eval.yaml`, build the badge
comment, and post it. Idempotent via a sentinel:
`<!-- ssb:badge --> sha={sha} path={path}` — if the discussion already contains it, the
badge is skipped (no duplicates). → [badges.md](badges.md)

### `preview-manifest.yml` — preview an Arena manifest PR
```yaml
on:
  workflow_dispatch:
    inputs:
      repo:   { description: "HF dataset repo containing manifest.yaml", required: true }
      pr:     { description: "HF PR/discussion number to comment on", required: true }
      branch: { description: "HF ref to preview, e.g. refs/pr/42", required: true }
```
Runs `speech-spoof-bench ci preview-manifest --repo … --pr … --branch …`. It downloads
the candidate `manifest.yaml`, compares its dataset IDs to `main`, walks every listed
dataset's verified submissions, and posts an Arena-like preview: dataset count, displayable
row count, warning count, added datasets, and removed datasets. Exit 0 only when the preview
has no warnings.

## Secrets & env vars — the complete table

| Name | Lives in | Used by | Purpose |
|------|----------|---------|---------|
| `HF_WEBHOOK_SECRET` | HF Space secrets | `arena/webhook.py` | Validate the inbound webhook (HMAC / `X-Webhook-Secret`). |
| `GH_PAT` | HF Space secrets | `arena/webhook.py` | GitHub token to dispatch the Actions. Needs **Actions: read+write** on `lab260ru/speech_spoof_bench`. |
| `GH_VERIFY_WORKFLOW_REPO` | HF Space env (optional) | `arena/webhook.py` | Target repo for dispatch. Default `lab260ru/speech_spoof_bench` (override for a fork). |
| `SPACE_COMMIT_TOKEN` | HF Space secrets | `arena/cache_store.py` | HF token to commit `cache.json` back to the Space. |
| `ARENA_SPACE_REPO` | HF Space env (optional) | `arena/cache_store.py` | Space repo id. Default `SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena`. |
| `HF_BOT_TOKEN` | GitHub Actions secret | `ci/verify_pr.py`, `ci/post_merge_badge.py`, `ci/nightly.py`, `ci/preview_manifest.py`, `hf_fetch.py` fallback | HF token to comment on dataset discussions and, when `HF_TOKEN` is absent, authenticate Hub downloads. |
| `GH_TOKEN` / `GITHUB_TOKEN` | GitHub Actions (built-in) | `ci/nightly.py` | The `gh` CLI for issue management. |
| `GH_RUN_URL` | GitHub Actions (composed) | `ci/verify_pr.py`, `ci/post_merge_badge.py` | Link back to the CI run in comments. Defaults to a hardcoded `lab260ru` URL if absent. |
| `HF_TOKEN` | local / runner env | `hf_fetch.py` | Preferred auth for downloading (private) HF artifacts. |
| `SSB_HF_TIMEOUT` / `SSB_HF_ATTEMPTS` / `SSB_HF_RETRY_SLEEP` | local / runner env | `hf_fetch.py` | Optional bounds for Hub downloads and metadata requests. Defaults: 20 s, 3 attempts, 1 s sleep. |
| `SSB_ARENA_MANIFEST_REPO` / `SSB_ARENA_MANIFEST_REVISION` | local / Space env | `manifest.py`, Arena ingest | Point a staging/local Arena at a manifest repo/ref other than production `main`. |

(See the project memory for which concrete tokens are provisioned — `project_phase8_secrets`.)

## Failure modes worth knowing

- **`HF_BOT_TOKEN` missing** → `verify_pr`/`post_merge_badge` *print* the verdict/badge to
  stdout instead of posting. Tests pass; nothing appears on HF. Easy to miss.
- **Any of `HF_WEBHOOK_SECRET` / `GH_PAT` / `SPACE_COMMIT_TOKEN` missing** → respectively:
  401 on every webhook, dispatch logged-and-skipped, cache never persists. All "soft"
  (logged, not crash).
- **New dataset added to the manifest** isn't routed by the webhook until the Arena does a
  `force_refresh` ingest (which any subscribed event triggers) — until then it's "not
  subscribed".
- **Hub/network stalls** are bounded for package metadata/file calls through `hf_fetch.py`
  (timeouts + retries). Raw third-party calls outside that wrapper still need review before
  being added to CI.
- `nightly.py` needs `gh` on the runner (present on `ubuntu-latest`) and uncaught `gh`
  failures will error the step.

## When CI changes require coordinated edits

- Rename/relocate the package repo → update `GH_VERIFY_WORKFLOW_REPO` default and the
  hardcoded `GH_RUN_URL` fallback in `verify_pr.py`.
- Rename a workflow file or change its `inputs` → update the dispatch URL + `inputs` dict
  in `webhook.py` (`_dispatch_verify_workflow` / `_dispatch_post_merge_workflow`).
- Change a `ci` CLI flag name → update the matching `run:` line in the `.yml`.
- Change the badge sentinel string → it must stay identical in `post_merge_badge.py`
  (write side) and `_already_posted()` (read side) or duplicates appear.
</content>
