# CI 429 / PR-burst follow-up (deferred side task)

**Status:** TODO — start *after* the RawTFNet submission PRs are merged. Do not
touch CI mid-submission.

Triggered while submitting RawTFNet (5 PRs at once): two distinct, real problems
surfaced beyond the already-documented "burst drops pending dispatches" residual
in `2026-06-03-ci-429-hardening.md`.

## Problem 1 — the queue can't handle many PRs
`concurrency: {group: hf-ci, cancel-in-progress: false}` (in `verify-hf-pr.yml`
and `post-merge-badge.yml`) is **global**. GitHub keeps only **1 running + 1
pending** per group and silently evicts older pending dispatches. A burst of >2
PR webhooks loses the extras with **no self-healing** — they never run until
manually re-fired (commit to `refs/pr/N`).

## Problem 2 — a single serialized run still 429s (observed, with traceback)
`verify_pr.py::_changed_submissions` calls
`api.list_repo_files(repo, ...)` **twice** (main + PR branch), each a **full
recursive tree listing of the whole dataset repo** (incl. all parquet data
shards). For ASVspoof2021_LA/DF that paginates many pages = many requests, just
to find which `submissions/*.yaml` was added. Observed: it 429'd on the *first*
call and blew through the retry budget.
- `_hf_retry.retry_on_429`: `_attempts=6, _cap=60`, no jitter → ~62 s total
  backoff (2+4+8+16+32). Against sustained account-level throttle that's too
  little, and **each retry re-paginates from scratch**, amplifying load.

## Fixes

### Problem 2 (do first — highest leverage, smallest change)
1. **Scope the listing to `submissions/`.** Replace the two full-tree
   `list_repo_files` calls with a path-scoped listing
   (`api.list_repo_tree(repo, path_in_repo="submissions", recursive=False,
   revision=…)`), main + branch. Turns many pages into one request each.
2. **Harden `retry_on_429`.** Raise `_attempts` (~8–10), raise `_cap`
   (~120–300 s), add **jitter**; keep honoring `Retry-After`.
3. Avoid redundant calls per run (one scoped listing; reuse where possible).
Package change ⇒ bump version (`pyproject` + `__init__`), keep schema/fixtures in
sync, and **bump the Arena pin** in `arena/requirements.txt` only if the Space
needs it (CI-only change may not). Add tests for the scoped listing + retry.

### Problem 1 — DONE (2026-06-04)
- **Kept** the global `hf-ci` serialization (primary defense against parallel
  429 storms — removing `concurrency` to "unlimit the queue" was rejected as a
  regression).
- **Added a scheduled self-healing sweep** — `speech-spoof-bench ci sweep`
  (`src/speech_spoof_bench/ci/sweep.py`) + `.github/workflows/sweep-pending-verifications.yml`
  (cron */15, own `hf-sweep` concurrency group, `workflow_dispatch` with
  `max`/`dry_run`). It lists open PRs on each manifest dataset, finds those with
  no `verify-pr` verdict comment, and re-dispatches `verify-pr` for up to `max`
  (default **1**) per run — so a dropped-burst backlog drains one at a time
  without re-creating the eviction. Read-only `--dry-run` for ops. 5 unit tests +
  a live dry-run; full suite green. Bumps 0.3.1 → 0.3.2. (Per-PR concurrency key
  considered & rejected: stops PRs evicting each other but reintroduces parallel
  429 risk.)

## Problem 3 — Arena `/webhook` got auto-disabled (observed 2026-06-03)
HF disabled the Space webhook `…speechantispoofingarena.hf.space/webhook` after 3
consecutive failed deliveries during the merge burst. **Precise cause (confirmed
by reading `arena/webhook.py`):** the handler already backgrounds the cache
refresh (`background.add_task(_refresh_and_commit, …)`), BUT it still makes
**synchronous network calls in the request path** before returning:
`_pr_num_from_merge_commit(repo, head_sha)` (an HF API fetch of the merge commit
to regex the `(#N)` PR number, webhook.py:200) and the GitHub dispatch
`_dispatch_post_merge_workflow` / `_dispatch_verify_workflow` (:202/:215/:222).
During the 429 throttle those inline calls hung/failed, so the webhook response
was slow/5xx'd → 3 strikes → disabled. (A Space cold-start would compound it.)
Same 429 root cause as Problems 1–2.

### Fixes
- **Space app (`arena/webhook.py`) — small, precise:** move `_pr_num_from_merge_commit`
  and the `_dispatch_*` calls **into the background task** too. The request path
  should do only: secret check → dedup → `background.add_task(...)` → return 200,
  with **zero HF/GitHub network calls inline**. Covered by `arena/tests/test_webhook.py`.

  **STATUS: DONE + deployed (2026-06-04).** `arena` commit `fa50e3e2` pushed to
  Space `main`: new `_verify_async`/`_post_merge_async` wrappers, request path
  makes no network calls and never 5xxs on a dispatch failure (acks 200 + logs);
  v3 merge ack is optimistic `refresh+post-merge`. Also bumped the
  `speech-spoof-bench` pin `159331d→91530ef` so the Space ingest gets the
  hardened 429 retry. Arena suite green (103).

  **Webhook re-enabled (2026-06-04):** done manually by the user via the HF
  Webhook Settings page (API `enable_webhook` 403'd — token lacked webhook
  scope). Verified post-deploy: Space `RUNNING`, `/webhook` POST without secret
  → 401 (route live, secret enforced, ack-fast). **Problem 3 fully closed.**
- **CI workflows (mitigations):**
  1. **Keep-alive cron** — scheduled GH Action pings the Space every ~10–15 min so
     webhooks never hit a cold start.
  2. **Webhook health / auto-re-enable cron** — scheduled Action checks
     `list_webhooks`, calls `enable_webhook` if disabled, and alerts.
  3. **CI-driven refresh backup** — post-merge workflow calls the Arena refresh
     endpoint directly (with retry) so merges refresh the board even if the
     webhook is down; 30-min TTL remains the last-resort fallback.
- **Immediate:** re-enable the webhook via `HfApi.enable_webhook(<id>)` — but it
  will likely re-disable under the next burst until the async-ack handler +
  keep-alive land, so pair the re-enable with at least the keep-alive cron.

## End-to-end validation (live burst test, 2026-06-04)
Submitted a deterministic random-baseline test model across **4 datasets at once**
(a real burst), then watched with **no manual verify re-dispatch**:
- **P3 confirmed:** the webhook acked all 4 deliveries and was **not** disabled.
- **P1 problem reproduced:** GitHub's `hf-ci` cap left 1 success + evicted 3
  (2 cancelled, 1 failed) — exactly the drop the sweep exists to fix.
- **P2 confirmed live:** the verify runs' listing call is scoped to
  `…/InTheWild/tree/main/submissions` (not the full repo tree); 8 hardened retries.
- **P1 sweep confirmed:** the sweep dispatched verify via `GITHUB_TOKEN` and
  recovered **all 4 dropped PRs to green**, one per run.
- All 4 test PRs closed; test model repo deleted (board untouched).

**Two bugs the live test caught (unit tests couldn't — they inject the dispatcher)
→ fixed in PR #4 (0.3.3):**
1. sweep workflow referenced `secrets.GH_PAT` (only in the arena Space) → empty
   token in CI → found drops but dispatched nothing. Fixed: built-in
   `GITHUB_TOKEN` + `permissions: actions: write`.
2. `run()` logged "dispatched" even when skipped and exited 0. Fixed: dispatch
   returns bool; candidates-but-zero-dispatched now exits non-zero (red CI).

**Residual / minor follow-ups:**
- **Sustained account-level 429** is the one thing client retry can't fully beat:
  during the heavy test session, even the scoped+retried calls 429'd in throttle
  windows (recovered after a cooldown). The fixes *minimize* the surface; they
  can't erase an account rate-limit.
- `sweep.run()`'s `fetch_manifest()` call isn't retry/try-wrapped, so a transient
  429 there crashes the whole sweep (observed once). Wrap it in `retry_on_429` +
  tolerate failure. (Small hardening; not yet done.)

## Verification (per testing-and-pitfalls matrix)
- `pytest` (package + arena) green; version bumped both files; schema/fixtures in
  sync; Arena pin bumped iff Space-relevant; re-run a burst of ≥3 PRs and confirm
  all verdicts post without manual re-dispatch.
