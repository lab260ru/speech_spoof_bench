# Phase 9 — Badge layer (design)

**Status**: design, awaiting plan
**Roadmap**: `docs/roadmap/ROADMAP.md` §"Phase 9 — Badge layer"
**Builds on**: Phase 8 CI/CD (`docs/specs/2026-05-23-phase-8-cicd-design.md`)

## Goal

After a submission PR is merged on a dataset repo, automatically post a follow-up
HF discussion comment containing a paste-ready `result.yaml`, a static shields.io
badge line, and a one-liner upload command. The submitter pastes the YAML into
their model repo under `.eval_results/<dataset-org>/<dataset-name>/result.yaml`
and the badge line into their README. The model page on HuggingFace then shows a
backlink badge pointing to the Arena.

**Done when**: a working `EER on ASVspoof2019_LA` shields badge is rendered on
`huggingface.co/<you>/random-baseline-asas` and click-throughs land on the
Arena.

## Architecture

Phase 8 wired:

```
HF webhook → arena/webhook.py
              ├─ refs/heads/main  → cache refresh
              └─ refs/pr/<n>     → dispatch verify-hf-pr.yml
```

Phase 9 adds one branch:

```
HF webhook → arena/webhook.py
              ├─ refs/heads/main + payload.pr_num set
              │       → cache refresh AND dispatch post-merge-badge.yml
              │         with {repo, pr, sha}
              ├─ refs/heads/main (no pr_num)
              │       → cache refresh only (existing)
              └─ refs/pr/<n>     → dispatch verify-hf-pr.yml (existing)

post-merge-badge.yml
   → speech-spoof-bench ci post-merge-badge --repo … --pr … --sha …
       1. fetch the submission YAML(s) changed in <sha>
       2. validate via submission.parse_submission
       3. badge.build_paste_comment(submission, …)
       4. HfApi.comment_discussion (HF_BOT_TOKEN)
```

**Invariant**: the badge value and the embedded `result.yaml` are both derived
from the same merged submission YAML in the same process. They cannot drift from
each other at emission time.

## Components

### `src/speech_spoof_bench/badge.py` (new)

Two pure string builders. No I/O. Both raise `BadgeError` on invalid input.

```python
class BadgeError(Exception): ...

def build_result_yaml(submission: dict, *, arena_url: str) -> str:
    """Render the paste-ready result.yaml string from a parsed submission dict.
    Validates output against schema/result.schema.json before returning."""

def build_paste_comment(submission: dict, *,
                        arena_url: str,
                        dataset_canonical_id: str,
                        primary_metric: str,
                        gh_run_url: str) -> str:
    """Render the full Markdown comment body. Embeds the sentinel marker
    `<!-- ssb:badge -->` for idempotency detection."""

def _color_for_eer(eer_percent: float) -> str:
    """<2.0 → brightgreen, <5.0 → green, <10.0 → yellow, else lightgrey."""
```

`dataset_canonical_id` is `"<org>/<name>"` of the dataset repo (matches the
submission's `dataset.id`). `primary_metric` is the first id in that dataset's
`eval.yaml` `metrics:` list, fetched by the caller. `arena_url` is the public
Arena Space URL; comes from a constant in `badge.py` (single source of truth,
overridable for tests).

### `src/speech_spoof_bench/schema/result.schema.json` (new)

JSON Schema (draft-07) for `result.yaml`. Strict (`additionalProperties: false`
everywhere). Required shape:

```yaml
schema_version: 1                          # const 1
system:
  name: <string>
  slug: <kebab-case>
  paper:
    arxiv_id: <string>
dataset:
  id: <org>/<name>
  revision: <string>                        # 7-40 hex
  split: <string>
scores:
  n_trials: <int >= 0>
  n_skipped: <int >= 0>
  <metric_id>: <number>                     # at least one besides n_trials/n_skipped
arena:
  url: <https url>
  system_url: <https url>
artifact:
  scores_url: <https://huggingface.co/.../resolve/<sha>/.eval_results/...>
```

No `submitter`, no `reproduction`, no `bench_version` — those are submission-side
concerns. Schema lives beside `submission.schema.json`.

### `src/speech_spoof_bench/ci/post_merge_badge.py` (new)

Mirrors `ci/verify_pr.py`.

```python
@dataclass
class Outcome:
    path: str            # submissions/<file>.yaml
    posted: bool
    notes: str           # "ok" | "skipped: already commented" | "error: …"

def run(*, repo: str, pr: int, sha: str,
        api: HfApi | None = None,
        gh_run_url: str | None = None) -> int
```

Algorithm:

1. List submission YAMLs added/modified in `<sha>` vs. `main^`. Reuse the same
   over-inclusive heuristic as `verify_pr._changed_submissions` (compare repo
   file listings).
2. For each path:
   - `hf_hub_download(repo_id=repo, filename=path, revision=sha, repo_type="dataset")`
   - `submission.parse_submission(text)` → dict
   - Fetch `eval.yaml` from the dataset (at the submission's pinned revision) and
     read `metrics[0]` as the primary metric. Cache per-revision in-process.
   - Build comment via `badge.build_paste_comment`.
   - Look up existing comments on the discussion. If any comment body contains
     the sentinel `<!-- ssb:badge --> sha=<sha> path=<path>`, skip.
   - Post via `api.comment_discussion`.
3. Return `0` if all submissions either posted or skipped cleanly; non-zero if
   any submission errored (e.g., primary metric missing from `scores`).

Token fallback identical to `verify_pr._post_comment`: missing `HF_BOT_TOKEN` →
print to stdout, return 0.

### CLI (`src/speech_spoof_bench/cli.py`)

Extend the existing `ci` subparser:

```bash
speech-spoof-bench ci post-merge-badge \
    --repo <org/dataset> \
    --pr <discussion-num> \
    --sha <merge-sha>
```

### Webhook (`arena/webhook.py`)

In the existing `refs/heads/main` branch:

- After the cache-refresh dispatch, inspect the HF webhook payload for the
  fields HF sets when the commit landed via a PR merge (`event.pull_request`
  number — exact field name confirmed against current HF webhook schema during
  implementation).
- If present, dispatch `post-merge-badge.yml` via the existing GitHub REST
  helper with `{repo, pr, sha}`.
- Self-event suppression (Space-authored commits via `SPACE_COMMIT_TOKEN`)
  applies before this branch runs — already in place from Phase 8b.

### `.github/workflows/post-merge-badge.yml` (new)

Structural copy of `verify-hf-pr.yml`. `workflow_dispatch` inputs
`{repo, pr, sha}`. Same `python:3.11` + `pip install -e .` + `HF_BOT_TOKEN`
secret. Same minimum-scope.

## Comment template

```markdown
**speech-spoof-bench** — submission merged ✅

System `<slug>` is now live on the [Arena](<arena_url>?system=<slug>).

To display a backlink badge on your model page, take the two steps below.

### 1. Add `result.yaml` to your model repo

```yaml
<result.yaml body>
```

Upload it with:

```bash
huggingface-cli upload <your-model-repo> result.yaml \
  .eval_results/<dataset-org>/<dataset-name>/result.yaml
```

### 2. Add the badge line to your README

```markdown
[![<METRIC> <VALUE> on <DATASET>](https://img.shields.io/badge/<METRIC>%20on%20<DATASET-encoded>-<VALUE>-<COLOR>)](<arena_url>?system=<slug>)
```

<!-- ssb:badge --> sha=<sha> path=<submission-path>

---
_🤖 [view CI run](<gh_run_url>)_
```

Notes on rendering:

- The shields URL uses `__` to mean a literal `_` in the dataset name segment.
  E.g. `ASVspoof2019_LA` → `ASVspoof2019__LA`.
- `%` in the value is URL-encoded to `%25`. E.g. `1.23%` → `1.23%25`.
- Badge color: `_color_for_eer(value)` for the `eer_percent` metric. For metrics
  added later, `badge.py` exposes per-metric color funcs; default `blue`.
- The sentinel `<!-- ssb:badge --> sha=<sha> path=<path>` is HTML-comment-stripped
  by HF's renderer and is what makes the post-merge step idempotent.

## Edge cases

| Case | Behavior |
|---|---|
| Merge touched no `submissions/*.yaml` | Skip silently; no comment. Same heuristic as `verify_pr._changed_submissions`. |
| Multiple submission YAMLs in the same merge | One comment per submission, each scoped to its dataset and path. |
| Merged YAML fails `submission.parse_submission` | Log warning, exit non-zero. Should not happen — `verify-pr` already gated this. |
| Primary metric absent from `scores` dict (mismatched `eval.yaml`) | Log error, no comment for that submission, exit non-zero so CI shows red. |
| `HF_BOT_TOKEN` missing | Print comment body to stdout, return 0. Symmetric with `verify_pr`. |
| Workflow re-runs against the same sha | Per-comment sentinel scan skips duplicates. Idempotent at the submission-path granularity. |
| Webhook payload has no PR metadata (commit landed directly on main) | Webhook does not dispatch the workflow. Direct-to-main is not a supported path. |
| Dataset's `eval.yaml` missing or `metrics:` empty | Log error, exit non-zero. Reflects a broken dataset; should already be blocked by `validate-dataset`. |

## Testing

### Unit (`speech-spoof-bench/tests/`)

| File | Asserts |
|---|---|
| `test_badge_build_result_yaml.py` | Golden snapshot of YAML string for a canonical submission dict. Catches key reordering / indentation drift. |
| `test_badge_color_for_eer.py` | Table of `(eer, color)` including boundaries `2.0`, `5.0`, `10.0`. |
| `test_badge_result_schema.py` | `result.schema.json` rejects: missing `arena.system_url`, extra top-level keys, wrong `schema_version`, non-https `arena.url`. |
| `test_badge_build_paste_comment.py` | Snapshot of full Markdown body. Includes sentinel marker. Verifies `%` → `%25` and `_` → `__` encoding in the shields URL. |
| `test_badge_primary_metric_missing.py` | `build_paste_comment` raises `BadgeError` when `primary_metric` is absent from `submission["scores"]`. |

### CI handler (mock HfApi, patterned on existing `test_verify_pr*.py`)

| File | Asserts |
|---|---|
| `test_post_merge_badge_no_submission_change.py` | Merge touched no `submissions/*.yaml` → `run()` returns 0, no `comment_discussion` call. |
| `test_post_merge_badge_multiple_submissions.py` | Two submission YAMLs in the same sha → two `comment_discussion` calls, each scoped correctly. |
| `test_post_merge_badge_idempotent.py` | Existing comment carries `<!-- ssb:badge --> sha=<sha> path=<path>` → `run()` skips that path, returns 0. |
| `test_post_merge_badge_no_token.py` | `HF_BOT_TOKEN` unset → prints to stdout, returns 0, no HfApi call. |
| `test_post_merge_badge_primary_metric_missing.py` | Submission missing the primary metric → `run()` returns non-zero, no comment posted for that path. |

### Webhook (`arena/tests/`)

| File | Asserts |
|---|---|
| `test_webhook_dispatches_post_merge.py` | Payload with `refs/heads/main` AND PR metadata → triggers `workflow_dispatch` for `post-merge-badge.yml` with `{repo, pr, sha}`. Existing cache refresh still runs. |
| `test_webhook_no_pr_no_dispatch.py` | Same payload but no PR metadata → only refresh runs; no badge dispatch. |
| `test_webhook_space_self_event_suppressed.py` | Self-event (Space-authored commit) → neither refresh nor badge dispatch. Already covered by Phase 8 test; extended assertion. |

### Manual end-to-end (the Phase 9 DoD)

1. Push a trivial re-submission PR for `random-baseline-asas` on the LA dataset
   repo (already deployed in Phase 6).
2. Merge it. Within ~60s a second bot comment appears on the merged discussion
   carrying the YAML + badge line + upload one-liner.
3. Copy `result.yaml` from the comment; run the upload one-liner against
   `<you>/random-baseline-asas`.
4. Paste the badge Markdown into the model repo README.
5. Visit `huggingface.co/<you>/random-baseline-asas`. Verify the badge renders
   and clicks through to the Arena.

## Out of scope

Deferred to Phase 10 or later:

- Arena reading `result.yaml` from model repos to verify the backlink exists
  (potential upgrade path to a `★` badge).
- Deep-link routing on `?system=<slug>` in the Arena tab logic (the slug is
  stable now; routing wires up in Phase 10).
- Dynamic badge endpoint served by the Arena.
- Multiple-metric badges; only the dataset's primary metric is rendered.
- Bot pushing the YAML / README edit directly to the submitter's model repo
  (would require submitter-granted write tokens).

## Docs follow-up

`docs/roadmap/ROADMAP.md` Phase 9 entry references "§3.6.2" of PLAN.md, which is
absent in the current PLAN.md. As part of this phase, add §3.6.2 to PLAN.md
documenting the `result.yaml` shape and its placement under
`.eval_results/<dataset-org>/<dataset-name>/result.yaml`, then update the
ROADMAP entry to point at the now-existing section.
