# Phase 8 — CI/CD Layer (Design)

**Date:** 2026-05-23
**Status:** Draft, awaiting user review
**Roadmap reference:** `docs/roadmap/ROADMAP.md` §Phase 8
**Spec reference:** `docs/roadmap/PLAN.md` §1.7, §3.3, §3.5, §6.4, §8

---

## 1. Goal

Make merged HF PRs flow automatically into the Arena — validated, reproduced, commented on, and reflected on the live Space without a maintainer hitting Refresh. Plus one cross-cutting addition: a local-dataset registry so maintainer-side reproducibility checks don't always stream from HF.

The phase ends when:
- Opening an HF PR on a subscribed dataset → CI comment with `schema + sha256 + EER` verdict appears within ~2 min.
- Merging that PR → Arena reflects the new row within ~60 s.
- Nightly cron detects 404 / sha drift / EER drift on previously-merged submissions and opens GitHub issues.
- `speech-spoof-bench reproduce --scoring` and friends can be pointed at a local parquet directory via a registry file, with zero per-call flag noise.

---

## 2. Architecture overview

Five deliverables, in dependency order:

1. **Local-dataset registry** — `~/.config/ssb/local.yaml`, consulted by `loader.resolve`.
2. **Arena → Docker Space** — FastAPI host serving Gradio at `/` and `/webhook`.
3. **`cache.json` persistence** — committed back to the Space repo for instant cold start.
4. **CI pipeline** — `speech-spoof-bench ci verify-pr` + `.github/workflows/verify-hf-pr.yml`.
5. **Event bridge + nightly** — HF webhook → either refresh+commit (main events) or `gh workflow run` (PR events); `nightly-revalidate.yml` cron walks all merged submissions.

Plus the secrets matrix (§8) and the manual runbook (§9).

```
┌─ HF dataset repo (LA) ──┐                      ┌─ GitHub repo (pip pkg) ─┐
│  PR opened ─────────────┼─► webhook ──► /webhook ──► workflow_dispatch ──┼─► verify-hf-pr.yml
│  main commit ───────────┼─► webhook ──► /webhook ──► refresh + commit    │   └─► ci verify-pr ──► HF discussion comment
└─────────────────────────┘            │ cache.json                        │
                                       ▼                                   │   nightly-revalidate.yml (cron 06:00 UTC)
                            ┌─ Arena Docker Space ─┐                       │     └─► ci nightly-revalidate ──► GH issues
                            │ FastAPI + Gradio     │                       │
                            └──────────────────────┘                       │
```

---

## 3. Local-dataset registry

### 3.1 Registry file

Path: `~/.config/ssb/local.yaml` (created lazily). Format:

```yaml
schema_version: 1
datasets:
  SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA: /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
```

The mapped path must be a directory containing `eval.yaml` and `data/test-*.parquet` (validated on `local set`, re-validated on each loader call — missing dir errors loudly, never silently falls back to HF).

### 3.2 Loader integration

In `loader.resolve(spec, *, streaming=True, force_remote=False)`:

1. If `spec` is a `Path` that exists → existing `_resolve_local` path.
2. **NEW:** if `spec` matches `<org>/<name>` and `force_remote` is False, consult `local.yaml`. If the id is mapped, dispatch through `_resolve_local(mapped_path, streaming)` (revision recorded as `None`).
3. Otherwise dispatch through `_resolve_hf`.

`reproduce.py` reads only `label` + `notes` columns via `select_columns`; that already works against local parquet, so no scoring-side changes are needed.

### 3.3 CLI

New subcommand group `local`:

```
speech-spoof-bench local set    <org/name> <path>   # validates + writes
speech-spoof-bench local list                       # prints current mapping
speech-spoof-bench local unset  <org/name>
speech-spoof-bench local show   <org/name>          # prints "local: <path>" or "remote"
```

Every command that hits a dataset gains a `--no-local` flag that sets `force_remote=True` for that invocation. Used to force a fresh HF stream (e.g. "did upstream change?").

### 3.4 Test surface

- Loader: round-trip `set/get/unset`; mapped path missing → raises with helpful message; `--no-local` overrides.
- Loader: registry absent → behavior identical to current code (regression guard).

---

## 4. Arena Docker upgrade (8a)

### 4.1 New file layout in `arena/`

```
arena/
├── Dockerfile          # NEW
├── main.py             # NEW — FastAPI host, mounts Gradio at /
├── webhook.py          # NEW — /webhook handler (§6)
├── cache_store.py      # NEW — cache.json read/write + HF commit-back (§5)
├── app.py              # existing — exports build_demo() unchanged
├── ingest.py           # existing — gains hydrate_from_cache(state) helper
├── ranking.py, schema.py
├── README.md           # frontmatter switches sdk → docker
├── requirements.txt    # + fastapi, uvicorn[standard], huggingface_hub already present
└── tests/              # existing tests untouched; new tests under tests/
```

### 4.2 `README.md` frontmatter

```yaml
---
title: Speech Anti-Spoofing Arena
sdk: docker
app_port: 7860
---
```

### 4.3 Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 4.4 `main.py`

```python
from fastapi import FastAPI
import gradio as gr
from app import build_demo
from webhook import router as webhook_router
import cache_store, ingest

app = FastAPI()
app.include_router(webhook_router)            # POST /webhook

@app.on_event("startup")
async def _startup():
    cached = cache_store.load()
    if cached is not None:
        ingest.hydrate(cached)
    # kick off background refresh; do not block startup

demo = build_demo()
app = gr.mount_gradio_app(app, demo, path="/")
```

One uvicorn process; `ArenaState` is shared in-process — no IPC.

### 4.5 Tests

Existing arena tests target `ingest`, `ranking`, `schema` directly (no Gradio launch) and continue to pass. CI gains a smoke `docker build` step (no `docker run`).

---

## 5. `cache.json` persistence (8b)

### 5.1 File format (committed to Space repo root)

```json
{
  "schema_version": 1,
  "loaded_at": "2026-05-23T12:00:00Z",
  "manifest_revision": "abc123",
  "rows":     [ /* serialized Row dicts */ ],
  "warnings": [ /* serialized Warning dicts */ ]
}
```

### 5.2 `cache_store.py` API

```python
def load() -> ArenaState | None: ...
def save_and_commit(state: ArenaState, *, reason: str) -> None: ...
```

- `load`: reads `./cache.json` if present, deserializes into `ArenaState`. Returns None on missing/corrupt (logged).
- `save_and_commit`:
  - Serialize → write `./cache.json`.
  - Compute content hash. If equal to last committed hash, skip.
  - If <30 s since last commit, defer (debounce).
  - Otherwise `HfApi(token=SPACE_COMMIT_TOKEN).upload_file(path_or_fileobj=..., path_in_repo="cache.json", repo_id=arena_space, repo_type="space", commit_message=f"cache refresh ({reason})")`.

### 5.3 Cold-start flow

1. `main.py` startup hook → `cache_store.load()` → if present, hydrate state. UI renders immediately.
2. Background task: `ingest.load_state(force_refresh=True)` → if state differs → `cache_store.save_and_commit(reason="startup-refresh")`.

### 5.4 Failure handling

- HF upload fails → log warning, keep in-memory state, retry on the next refresh event.
- Corrupt `cache.json` → log + treat as missing; rebuild from scratch.

---

## 6. Webhook handler (8c, 8e)

### 6.1 Endpoint

`POST /webhook` — implemented in `arena/webhook.py` as a `fastapi.APIRouter`.

### 6.2 Authentication

HF webhook sends `X-Webhook-Secret` header (configured per-webhook in HF UI). Compare with `HF_WEBHOOK_SECRET` env var using `hmac.compare_digest`. 401 on mismatch.

### 6.3 Payload routing

```
event.scope = repo.content, ref = refs/heads/main, repo in subscribed set
  → background: ingest.load_state(force_refresh=True)
                cache_store.save_and_commit(reason=f"main@{repo}")
  → return 200

event.scope = repo.content, ref starts with refs/pr/  (or discussion.isPullRequest)
  → POST GitHub workflow_dispatch (see 6.4)
  → return 200

anything else
  → log and return 200
```

Subscribed datasets = manifest `core_set + extended` ids at handler load time. Refreshes when manifest is re-fetched.

### 6.4 GitHub workflow dispatch

Direct REST call (no `gh` binary in the Docker image):

```
POST https://api.github.com/repos/SpeechAntiSpoofingBenchmarks/speech-spoof-bench/actions/workflows/verify-hf-pr.yml/dispatches
Authorization: Bearer <GH_PAT>
Content-Type: application/json
{
  "ref": "main",
  "inputs": {"repo": "<dataset-id>", "pr": "<n>", "branch": "<ref>"}
}
```

### 6.5 Idempotency

In-memory LRU of `(repo, newSha)` seen in the last 5 min; duplicates from HF redeliveries are dropped.

### 6.6 Tests

- Bad secret → 401.
- Main event on subscribed repo → schedules refresh (mocked `ingest`).
- PR event → POSTs to GH dispatch (mocked `requests.post`).
- Unsubscribed repo → 200 + log skip; no state mutation.

---

## 7. `ci verify-pr` CLI + `verify-hf-pr.yml` (8d)

### 7.1 CLI

```
speech-spoof-bench ci verify-pr \
    --repo <dataset-id> --pr <n> --branch <ref>
```

Behavior:

1. `HfApi.list_repo_files(repo_id=..., revision=branch)` → diff `submissions/*.yaml` against `main`. Added or modified files only; deletions skipped.
2. For each changed YAML:
   - Download file from PR branch.
   - `validate-submission` (schema check, offline).
   - `reproduce --scoring` (fetches `scores_url`, sha-verifies, recomputes EER vs claimed within `1e-6`).
3. Build a markdown table:

   ```
   | Submission | Schema | sha256 | EER match | Notes |
   ```

4. Post comment on the PR's HF discussion via `HfApi(token=HF_BOT_TOKEN).comment_discussion(...)`. Comment footer: `🤖 speech-spoof-bench ci verify-pr — <link to GH run>`.
5. Exit 0 iff every row passes; otherwise 1 (so the GH workflow shows red on the PR).

A failure during scoring reproduction (e.g. 404 on `scores_url`) becomes a row with the error in Notes — the comment still posts. Only a *complete* failure (cannot post comment) leaves the workflow red with no artifact.

### 7.2 `.github/workflows/verify-hf-pr.yml`

```yaml
name: verify-hf-pr
on:
  workflow_dispatch:
    inputs:
      repo:   {required: true, type: string}
      pr:     {required: true, type: string}
      branch: {required: true, type: string}
jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -e .
      - run: |
          speech-spoof-bench ci verify-pr \
            --repo   "${{ inputs.repo }}" \
            --pr     "${{ inputs.pr }}" \
            --branch "${{ inputs.branch }}"
        env:
          HF_BOT_TOKEN: ${{ secrets.HF_BOT_TOKEN }}
```

### 7.3 Tests

Mock `HfApi` PR diff with one good and one bad submission → exit 1, comment markdown matches snapshot.

---

## 8. Nightly revalidate (8f) + secrets matrix (8g)

### 8.1 `.github/workflows/nightly-revalidate.yml`

```yaml
name: nightly-revalidate
on:
  schedule: [{cron: "0 6 * * *"}]   # 06:00 UTC daily
  workflow_dispatch:
jobs:
  walk:
    runs-on: ubuntu-latest
    permissions: {issues: write}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -e .
      - run: speech-spoof-bench ci nightly-revalidate --open-issues
        env:
          HF_BOT_TOKEN: ${{ secrets.HF_BOT_TOKEN }}
          GH_TOKEN:     ${{ secrets.GITHUB_TOKEN }}
```

### 8.2 `ci nightly-revalidate` CLI

1. `fetch_manifest()` → all dataset ids.
2. For each id → `list_submission_files` → for each `submissions/*.yaml`:
   - HEAD `scores_url` (must be 200; must be a `/resolve/<sha>/` URL, not `/main/`).
   - Recompute sha256 of fetched bytes vs `scores_sha256`.
   - `reproduce --scoring` (EER within `1e-6` of YAML claim).
3. For each failure: find an existing `stale-submission`-labelled issue with title `[<dataset>] <slug>`. If none → `gh issue create`. If one and the failure detail differs from the last comment → `gh issue comment`.
4. On clean pass, close any matching open issue.

Sequential walk; ~seconds per submission; whole walk on ≤10 submissions <5 min. Free GH Actions on public repos.

### 8.3 Secrets matrix

| Secret | Lives | Scope | Used by |
|---|---|---|---|
| `HF_WEBHOOK_SECRET` | Arena Space env + each HF webhook config | shared secret, no API perms | `/webhook` HMAC check |
| `SPACE_COMMIT_TOKEN` | Arena Space env | HF token, write on `spaces/SpeechAntiSpoofingBenchmarks/arena` only | `cache_store.save_and_commit` |
| `HF_BOT_TOKEN` | Arena Space env + GH repo secret | HF org bot token, write on `datasets/SpeechAntiSpoofingBenchmarks/*` (discussions if HF supports finer) | `comment_discussion` from CI + nightly |
| `GH_PAT` | Arena Space env | GitHub fine-grained PAT, `actions:write` on `SpeechAntiSpoofingBenchmarks/speech-spoof-bench` only | Webhook → workflow dispatch |

Built-in `GITHUB_TOKEN` handles nightly issue ops in-repo; no PAT needed there.

---

## 9. Manual runbook

These steps cannot be automated and will be presented as a checklist after each implementation slice.

1. **Tokens.** Create the four secrets with minimum scopes (§8.3). Save real values into HF Space secrets + GH repo secrets. Commit a `.env.example` with placeholders.
2. **Docker Space conversion.** Edit `arena/README.md` frontmatter to `sdk: docker` and push; if HF does not switch SDK, delete the Space in the UI and recreate as Docker (same name) before pushing.
3. **Local docker run.** `docker build -t ssb-arena arena/` → `docker run -p 7860:7860 ssb-arena` → confirm UI matches the live Gradio Space.
4. **HF webhook.** Dataset repo `ASVspoof2019_LA` → Settings → Webhooks → add `https://<space-url>/webhook` with `HF_WEBHOOK_SECRET`, subscribed to `repo.content`. Push a no-op commit → verify cache refresh and a new `cache.json` commit on the Space repo.
5. **PR end-to-end.** Open a hand-crafted PR on LA that bumps `eer_percent` by +5. Watch the GH workflow fire; the HF discussion comment must appear within ~2 min and report EER mismatch. Push a revert → comment updates green.
6. **Nightly.** `gh workflow run nightly-revalidate.yml` manually → clean run. Temporarily corrupt a `scores_sha256` → next run opens a `stale-submission` issue. Revert → next run closes it.
7. **Local-dataset registry.** `speech-spoof-bench local set SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA` → `speech-spoof-bench reproduce --scoring submissions/random-baseline.yaml` against the LA dataset repo checkout → confirm labels are read from the local parquet (visible in INFO logs) and EER matches.

---

## 10. Staged delivery order

Each slice is independently mergeable and testable:

1. Local-dataset registry (CLI + loader + tests) — ship, manually verify.
2. Arena Docker (Dockerfile, `main.py`, mount Gradio) — local docker run, then push to HF.
3. `cache.json` (`cache_store.py`, startup hydration, commit-back).
4. Webhook handler — manually verify refresh-on-commit before adding GH bridge.
5. `ci verify-pr` CLI — testable offline first.
6. `verify-hf-pr.yml` + webhook → GH bridge — end-to-end PR test.
7. `nightly-revalidate.yml` — last; lowest risk.

---

## 11. Test surface summary

| Area | Test |
|---|---|
| Loader + registry | round-trip set/get/unset; missing mapped path errors loudly; `--no-local` forces HF code path. |
| `cache_store` | serialize → deserialize equality; debounce on unchanged content. |
| `webhook` | bad secret → 401; main event → schedules refresh (mocked); PR event → POSTs to GH dispatch (mocked). |
| `ci verify-pr` | mock PR diff with good + bad submission → exit 1, comment text matches snapshot. |
| `ci nightly-revalidate` | mock manifest + submissions; broken sha → opens issue; clean rerun → closes issue. |
| Docker | `docker build` smoke step in CI. |

---

## 12. Out of scope

- `reproduce --inference` (still raises `NotImplementedError`; lands in Phase 12+ when GPU compute is available).
- Per-discussion auto-merge or auto-close behavior — maintainer still merges manually.
- Caching audio locally (only labels are read; audio is not needed for scoring reproduction).
- Migrating from free HF Spaces to paid tier; persistent volume is *not* assumed.
