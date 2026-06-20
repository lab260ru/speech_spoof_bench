# How ranking works & what the badges mean

A short, plain-language guide to reading the leaderboard: what the tiers are, how the
**Mean** and **Rank** are computed, and what each badge tells you. No prior knowledge
needed.

## The one-sentence version

Every model is scored on a fixed list of datasets (the **Core Set**). How *many* of those
datasets a model covers decides its **tier**; how *well* it scores decides its **Rank**
within the board.

## The score: EER

The headline metric is **EER (Equal Error Rate), as a percentage**. **Lower is better** —
0% is perfect, ~50% is random guessing. It measures how often the detector confuses real
speech with fakes (and vice versa) at its best single threshold.

## Tiers — how much of the Core Set you cover

A model is sorted into a tier by its **coverage**: the share of Core-Set datasets it has a
verified score on.

| Tier | Coverage of the Core Set |
|---|---|
| 🥇 **Gold** | Full coverage — a score on every Core dataset. |
| 🥈 **Silver** | At least half of the Core datasets. |
| 🥉 **Bronze** | At least one Core dataset. |
| 🔓 **Unpublished / Proprietary** | Any coverage, but **no paper** (see below). |

Tiers reward *breadth*: a model that proves itself across many datasets is more
trustworthy than one tuned to a single benchmark. The exact thresholds live in the
`arena-manifest` and can change, so the board shows the current definition next to each
tier.

## The paper rule (why a model can be "unranked")

A **paper** is what lets a model compete in the ranked tiers (🥇/🥈/🥉). Proprietary or
unpublished models are welcome, but they appear in the dedicated **🔓 Unpublished /
Proprietary** tier and are left **unranked** there — no matter how strong their scores.
Add a paper later and the model moves into the ranked tiers. (Within the Unpublished tier
the rows are still sorted by score, so you can compare them, but they don't take a
board-wide Rank.)

## Mean & Rank — two ways to average

The **Mean** column is the single score that drives **Rank**, and it changes with the
**Ranking view** selector at the top of the Overview tab:

- **Aggregated** — a plain average of a model's EER over **only the Core datasets it
  actually ran**. Datasets it did not run are simply left out of the average — they are
  **not** penalised here (how much of the Core Set a model covers is shown instead by its
  tier: 🥇 Gold = full coverage, etc.). Every dataset counts equally, regardless of size.
- **Pooled** — a size-weighted average over the **whole** Core Set, where each dataset is
  weighted by how many audio clips (trials) it has, so **larger datasets have more
  impact**. Any Core dataset a model **never ran counts as 50% EER** (chance level) — so
  skipping a dataset, especially a large one, lowers the pooled score.

In both views a model's per-dataset cell stays **blank** for datasets it did not run; the
50% EER substitution feeds only the pooled Mean/Rank — it is never shown as a real result.

## Badges — what's been checked

Next to a model you'll see small badges:

- **✔ scoring** — a maintainer re-downloaded the model's score file, confirmed it is
  byte-for-byte unchanged (using a SHA-256 checksum — a fingerprint that changes if even
  one character is edited), and recomputed the metric. This is the standard verification
  every merged row carries.
- **★ inference** — a maintainer re-ran the model *end-to-end* from its checkpoint. This
  is a stronger, optional check. **It is not yet automated**, so today's verified rows
  carry **✔ scoring**.
- **📄 paper** — the model links a paper, so it can compete in the ranked tiers.

## Live tier & rank badges (for your model card)

The Arena also serves *live* badges you can paste into your model's README (the small
status images you often see at the top of a repo). They call the board and always show the
**current** standing, so they never go stale. Put your model's **slug** — its short id,
lowercase-with-hyphens — in the URL:

```
https://…hf.space/badge/<your-slug>/tier.json     → e.g. "gold"
https://…hf.space/badge/<your-slug>/rank.json     → e.g. "#1 of 11"
```

Both badges read **`unranked`** for any model the board can't rank — including
**unpublished / no-paper models**, which still show in their tier on the leaderboard but
don't take a board-wide rank. So the tier badge shows 🥇/🥈/🥉 only for ranked (papered)
models. For ready-to-paste badge snippets, see the
[badges reference](https://github.com/lab260ru/speech_spoof_bench/blob/main/docs/architecture/badges.md).

## In short

- **Tier** = how much of the Core Set you cover (breadth).
- **Rank / Mean** = how well you score (depth), in either the per-dataset-equal
  (aggregated) or size-weighted (pooled) view.
- **No paper** ⇒ Unpublished tier, unranked.
- **✔** is the verification every row gets today; **★** is the not-yet-automated upgrade.
