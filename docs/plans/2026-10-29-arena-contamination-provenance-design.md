# Arena "trained on" (contamination) via corpus provenance — design

> **Note:** Matching rule superseded later the same day by "same data only" — see `docs/architecture/provenance.md`.
Date: 2026-10-29 · Status: approved (maintainer); implemented in arena-live (local), not deployed · Scope: Arena (`arena-live/`) + arena-manifest repo

## Goal

Flag leaderboard cells where a model was trained on data that overlaps the test
dataset, generically: datasets declare the **source corpora** they are built from,
models declare what they **trained on**, and any overlap (rule C below) marks the
cell. No penalty.

## Behaviour of a flagged cell (already built in the local preview)

- Overview cell shows muted **n/a\***; tooltip names the overlapping corpus and the real score.
- Left out of the Mean / rank in both ranking views (`ranking.global_scores`); **no penalty**.
- Still counts toward coverage / tier.
- Never highlighted as best (green = best across all open-source tiers pooled; blue = best within the proprietary tier).
- Legend line explains n/a\*. Per-dataset tab marks the row's System with \*.

## Data: `provenance.yaml` in the arena-manifest repo (optional file)

```yaml
version: 1
corpora:                      # shared vocabulary
  VCTK: {}
  LibriSpeech: {parts: [train, dev, test]}
  LibriTTS: {from: LibriSpeech, parts: [train, dev, test]}
  ASVspoof2019-LA: {from: VCTK, parts: [train, dev, eval]}
datasets:                     # Arena dataset id -> source entries (bona fide audio + spoof speakers/references)
  SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA: [ASVspoof2019-LA:eval]
  SpeechAntiSpoofingBenchmarks/SONAR: [LibriTTS]          # no split -> whole corpus
models:                       # for systems that don't declare training data themselves
  "*-antideepfake*":          # fnmatch on slug
    trained_on: [ASVspoof2019-LA:train, ASVspoof2019-LA:eval, LibriTTS, VCTK, ...]
    source: https://arxiv.org/abs/2506.21090
```

Entry syntax: `Corpus` (whole corpus) or `Corpus:part` (`part` must be listed in that corpus's `parts`).

## Matching rule (option C: corpus-level, official splits respected)

1. **Lineage**: each entry expands up its `from` chain as (corpus, part) pairs. A
   `from: Parent:part` link records which split of the parent a derived corpus comes
   from (e.g. `ASVspoof2021-LA: {from: "ASVspoof2019-LA:eval"}`).
2. Two entries **overlap** iff their lineages share a corpus, judged at the **closest
   shared corpus only**: if both reach it through different parts, they are disjoint and
   ancestors above it (e.g. VCTK) don't count. Otherwise they overlap.
3. Cell (system S, dataset D) is flagged iff any training entry of S overlaps any source
   entry of D. Unknown split ⇒ whole corpus (conservative). Tooltip shows the shared corpus.

Examples: `ASVspoof2019-LA:train` vs `ASVspoof2021-LA` → clean (train vs eval at ASVspoof2019-LA).
`VCTK` vs `ASVspoof2019-LA:eval` → flagged (VCTK). `LibriTTS:train` vs `LibriTTS:test` → clean.
`Common-Voice:ru` vs `Common-Voice:en` → clean (languages modelled as disjoint parts).

Data conventions (v1): datasets list corpora of bona fide audio + cloned/converted speakers only
(not text-only reuse, not generator training data); LibriVox-derived corpora kept separate;
datasets without a public source corpus are omitted.

Result on the live board (2026-10-29): the 14 AntiDeepfake models are flagged on ODSS
(VCTK, Google-LatAm-Spanish) and XMAD (AISHELL-3, Common-Voice); nothing else.

## Sources of a model's training entries (union)

- `provenance.yaml` `models:` entries whose pattern matches the slug.
- A submission's optional `system.trained_on` (existing schema: Arena dataset ids):
  each id contributes that dataset's `datasets:` source entries (or the dataset id itself
  as an opaque corpus if it has none).

## Components

- `provenance.py` (replaces the preview's `contamination.OVERRIDES` dict):
  - `fetch_provenance()` — HF download from arena-manifest (cached per refresh);
    `ARENA_PROVENANCE_PATH` = local file override (used by `dev_local.sh`); missing file ⇒ empty.
  - `parse(text) -> (Provenance, warnings)` — unknown corpus / undeclared part / `from`
    cycle ⇒ warning + entry dropped; never raises.
  - `contaminated(rows, prov) -> {slug: {dataset_id: [overlapping corpora]}}` — pure.
- Ingest stores the parsed provenance on `ArenaState` (serialized in `cache.json`, so a cold start
  needs no fetch); parse warnings join `ArenaState.warnings`.
- `leaderboard.py` / `ranking.py` / `app.py` consume the `contaminated` map (already wired for the preview).

## Rollout

1. Commit `provenance.yaml` (corpora + all 24 Core datasets + AntiDeepfake models) to arena-manifest.
2. Deploy Arena code (together with the green/blue best-value highlighting) via HF upload.
3. Next refresh applies it. Code without file = no marks; file without code = ignored.

## Testing

Unit: expansion + split rules, derivation chains, unknown-split fallback, slug patterns,
`system.trained_on` mapping, parse warnings, missing file. Existing: n/a cell, Mean exclusion,
highlight exclusion, legacy cache load. Manual: local preview with the real `provenance.yaml`.

## Out of scope

Penalties; submitter-declared corpus names in `system.trained_on` (needs a package schema change);
editing the manifest schema.

## Data inputs

AntiDeepfake training data: arXiv 2506.21090 + github.com/nii-yamagishilab/AntiDeepfake
`protocols/README.md` — trains on train **and eval/test** splits of most sets (dev = model selection only);
held out: In-the-Wild, DEEP-VOICE, ADD2023 1.2 R2, FakeOrReal, Deepfake-Eval-2024.
Dataset source corpora: researched 2026-10-29, recorded with citations in `arena-manifest/provenance.yaml` comments.
