# Badges

There are **two kinds of badge**, and the difference matters:

1. **Static badge** — a fixed shields.io URL baked into a `result.yaml`/model card. Colour
   reflects the EER *at submission time*. Built by the package's `badge.py`.
2. **Dynamic badge** — a live shields.io *endpoint* badge that calls the Arena and shows
   the system's **current** tier or rank. Served by `arena/badges.py`. Never goes stale —
   if the board re-ranks, the badge updates.

## Static badges — `badge.py` (in the package)

Used by the **post-merge-badge** CI step to build the paste-ready comment a contributor
copies into their model card.

- `build_result_yaml(submission, arena_url)` → the `schema_version: 1` result projection
  (validated against `result.schema.json`). See [submission-lifecycle.md](submission-lifecycle.md#3-the-result-file-the-badge-layer).
- `build_paste_comment(...)` → the full Markdown comment: a "submission merged ✅"
  heading, the embedded `result.yaml` code block, `huggingface-cli upload` instructions
  (it goes to `<model-repo>/.eval_results/<dataset-id>/result.yaml`), the badge Markdown
  lines, the CI-run link, and the idempotency sentinel `<!-- ssb:badge --> sha=… path=…`.

### EER colour thresholds (`_color_for_eer`)

| EER (`eer_percent`) | shields colour |
|---------------------|----------------|
| `< 2.0` | `brightgreen` |
| `< 5.0` | `green` |
| `< 10.0` | `yellow` |
| `>= 10.0` | `lightgrey` |

Any **non-EER** metric defaults to `blue` (informational — it isn't tiered). `_fmt_value`
rounds to 2 decimals and strips trailing zeros.

## Dynamic badges — `arena/badges.py`

Two FastAPI routes returning shields.io-compatible JSON:

- `GET /badge/{slug}/tier.json`
- `GET /badge/{slug}/rank.json`

```json
GET /badge/random-baseline/tier.json
{ "schemaVersion": 1, "label": "arena tier", "message": "gold",    "color": "#FFD700" }

GET /badge/random-baseline/rank.json
{ "schemaVersion": 1, "label": "arena rank", "message": "#1 of 5", "color": "#FFD700" }

GET /badge/unknown-slug/tier.json
{ "schemaVersion": 1, "label": "arena tier", "message": "unranked", "color": "lightgrey" }
```

`_standing(slug)` loads the current `ArenaState`, calls `assign_tiers()` + `global_rank()`
(using the manifest's `default_view`), and returns `{tier, place, out_of, color}`. Errors
are swallowed → the badge shows `unranked` rather than 500ing. Each response sets
`Cache-Control: max-age=300` (shields caches for 5 min).

### Tier colours (`_tier_color`)
Looks up the tier's `color` in the manifest first; if absent, falls back to a positional
palette:

```
_PALETTE = ["#FFD700",  # gold
            "#C0C0C0",  # silver
            "#CD7F32",  # bronze
            "#4C9AFF",
            "#6554C0"]
```
Unranked → `lightgrey`.

## Using a dynamic badge in a model card

```markdown
[![arena tier](https://img.shields.io/endpoint?url=https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/<slug>/tier.json)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=<slug>)
[![arena rank](https://img.shields.io/endpoint?url=https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/<slug>/rank.json)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=<slug>)
```

`<slug>` is the system slug (`^[a-z0-9][a-z0-9-]*$`). The link target is the Arena's
per-system deep link, `?system=<slug>` — the same `arena.system_url` written into the
result projection.

## Gotchas

- A slug typo passes schema validation but produces a wrong badge link and an `unranked`
  dynamic badge — the system simply won't be found.
- The static EER colour is a *snapshot*; the dynamic tier/rank is *live*. Prefer the
  dynamic ones in a model card so the badge stays honest as the board grows.
- The sentinel string must match byte-for-byte between the writer (`post_merge_badge.py`)
  and the duplicate-check (`_already_posted`), or merges post duplicate badge comments.
</content>
