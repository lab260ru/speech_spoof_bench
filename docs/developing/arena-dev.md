# Developing the Arena

The Arena (`arena/`) is the HF Docker Space: a FastAPI process hosting the Gradio
leaderboard, the webhook, and the badge endpoints. It is **untagged** — it ships from
whatever is on the Space's `main` — and its only real "version" is the package commit it
pins in `requirements.txt`.

## Run it locally

```bash
cd ~/speech-spoof-bench/arena
pip install -e ".[dev]"          # or: pip install -r requirements.txt
pytest                            # ingest/ranking/leaderboard/charts/webhook/badges tests

# run the full app (FastAPI + Gradio + webhook + badges):
uvicorn main:app --host 0.0.0.0 --port 7860
# open http://localhost:7860  → UI;  http://localhost:7860/healthz → health
```

The UI hydrates from the committed `cache.json` instantly, then a background refresh hits
HF. For pure UI work you can iterate against the cached state without network.

## Module responsibilities (where to make a change)

| You want to change... | Edit |
|------------------------|------|
| A tab's layout / wiring | `app.py` |
| Table columns / sticky columns / HTML | `leaderboard.py` (`_PINNED_WIDTHS = [44, 150]`) |
| The ranking maths (tiers, gamma, penalty) | **`ranking.py`** — but most of this is *manifest data*, not code |
| Chart data (size scatter, SOTA timeline) | `charts.py` |
| What gets ingested / filtered | `ingest.py` (keep `_fetch_*` indirection for tests) |
| Cache format / commit behaviour | `cache_store.py` |
| Webhook routing / dispatch | `webhook.py` (see [../architecture/cicd.md](../architecture/cicd.md)) |
| Badge endpoints / colours | `badges.py` (see [../architecture/badges.md](../architecture/badges.md)) |
| Activity feed | `events.py` + `changelog.py` |

**Pure vs impure:** `leaderboard.py`, `ranking.py`, `charts.py`, `events.py` are pure
(DataFrames / dicts in → out, no I/O) — unit-test them directly with fixture `Row`s.
`ingest.py`, `cache_store.py`, `webhook.py`, `badges.py` do I/O — tests monkeypatch the
HF/GitHub calls. Don't move logic across that line without moving the tests.

## Tweaking ranking — do it in the manifest, not the code

Most "change the ranking" requests are **data** changes in `arena-manifest/manifest.yaml`,
not code: tier thresholds/colours, `requires_paper`, `gamma_aggregated`/`gamma_pooled`,
`absence_penalty`, per-dataset `weights`, `default_view`. Change those there and bump
`ranking_version` — no Arena code change, no package release. Only touch `ranking.py` if
you're changing the *formula itself*. See
[../architecture/arena.md](../architecture/arena.md#the-ranking-algorithm--rankingpy) and
[../architecture/versioning.md](../architecture/versioning.md).

## Environment variables (local vs Space)

The Space sets these as **Secrets**; locally you only need the ones for the path you're
exercising:

| Var | Needed for | If unset locally |
|-----|-----------|------------------|
| `HF_WEBHOOK_SECRET` | testing `/webhook` | webhook returns 401 |
| `GH_PAT` | dispatching real Actions | dispatch logged-and-skipped (fine for UI work) |
| `SPACE_COMMIT_TOKEN` | committing `cache.json` | state stays in memory only (fine locally) |
| `GH_VERIFY_WORKFLOW_REPO` | overriding dispatch target | defaults to `lab260ru/speech_spoof_bench` |
| `ARENA_SPACE_REPO` | overriding the Space repo id | defaults to the org Space |

For UI/ranking work you typically need **none** of them — the cached state renders and the
absent secrets just disable the live side effects.

## Testing the webhook locally

```bash
# with HF_WEBHOOK_SECRET set, simulate a PR-update event:
curl -X POST localhost:7860/webhook \
  -H "X-Webhook-Secret: $HF_WEBHOOK_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"event":{"scope":"repo.content"},
       "repo":{"name":"SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA","type":"dataset","headSha":"abc"},
       "updatedRefs":[{"ref":"refs/pr/42","newSha":"abc"}]}'
# → {"status":"scheduled","kind":"verify-pr"}  (dispatch is skipped without GH_PAT)
```

The `tests/test_webhook.py` fixtures cover the real payload variants (PR update, main
merge with/without `updatedRefs`, the v3 PR-number-from-commit-title recovery) — read them
to understand routing.

## Deploying

1. **Bump the package pin** in `requirements.txt` if the package changed:
   `speech-spoof-bench @ git+https://github.com/lab260ru/speech_spoof_bench.git@<new-sha>`.
   This is the project's recorded gotcha — skip it and the Space runs stale code, and the
   Submit tab shows stale guides (`docs_fetch` reads this SHA).
2. `pytest` green.
3. Push `arena/` to the Space remote
   (`huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena`). HF
   rebuilds the Docker image and restarts.
4. **Visually verify all tabs on the live Space** (Overview, Datasets, Per dataset, By
   model size, Over time, Submit, About) — the roadmap's standing manual check. UI
   regressions don't show up in unit tests.
5. Confirm `/healthz` is green and a badge endpoint responds:
   `…hf.space/badge/random-baseline/tier.json`.

## Cold-start & cache notes

- `cache.json` is committed back into the Space by the running app
  (`SPACE_COMMIT_TOKEN`). Don't hand-edit it expecting it to stick — a refresh overwrites it.
- The self-loop guard (`repo_type == "space"` ignored in the webhook) is what stops the
  cache commit from triggering another refresh. If you change webhook filtering, preserve it.
- Ingest TTL is 30 min (60 s on failure); a freshly merged submission appears within a
  refresh cycle, not instantly.

For the deeper architecture, read [../architecture/arena.md](../architecture/arena.md).
</content>
