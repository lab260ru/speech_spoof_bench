# Contributing to speech-spoof-bench

Thanks for helping build an open, reproducible leaderboard for speech anti-spoofing.
This page points you to the right guide for whatever you want to do, and lists the small
set of checks that keep a change from breaking once it's live.

## First, the golden rule

**Prove every layer locally before it touches Hugging Face or GitHub.** Most
"it worked while I was developing" failures come from skipping the offline loop and only
discovering the break once it's live. Start with
[developing/setup.md](docs/developing/setup.md) to get a fast, fully-offline inner loop.

## Pick your task

| I want to… | Read |
|---|---|
| Submit a model | [docs/submitting/submit-model.md](docs/submitting/submit-model.md) |
| Submit a dataset | [docs/submitting/submit-dataset.md](docs/submitting/submit-dataset.md) |
| Set up local dev | [docs/developing/setup.md](docs/developing/setup.md) |
| Add a model wrapper | [docs/developing/new-model.md](docs/developing/new-model.md) |
| Build a dataset | [docs/developing/new-dataset.md](docs/developing/new-dataset.md) |
| **Add a metric** *(good first PR)* | [docs/developing/new-metric.md](docs/developing/new-metric.md) |
| Work on the package | [docs/developing/contributing-package.md](docs/developing/contributing-package.md) |
| Work on the Arena | [docs/developing/arena-dev.md](docs/developing/arena-dev.md) |
| Avoid the silent breakages | [docs/developing/testing-and-pitfalls.md](docs/developing/testing-and-pitfalls.md) |

## Good first PRs

**Adding a metric** (say min-tDCF, accuracy, or an AUC) is the smallest "contribute to the
package" task and a good first PR — it's a self-contained plugin with a clear test.
See [docs/developing/new-metric.md](docs/developing/new-metric.md).

## Before you open a PR

Run the checks that match your change (the full matrix is in
[testing-and-pitfalls.md](docs/developing/testing-and-pitfalls.md)). The minimum bar:

- **`pytest` is green** — the tests are the contract.
- For a **package change**: the version is bumped in **both** `pyproject.toml` and
  `src/speech_spoof_bench/__init__.py`; if you changed a schema, bump its version `const`
  and update the test fixtures in the same PR (see
  [versioning.md](docs/architecture/versioning.md)); the Arena's pin
  (`arena/requirements.txt`) is bumped if the Space needs the new behaviour.
- For a **dataset**: `validate-dataset` is all-green offline, then green online.
- For a **manifest change**: it still validates against the schema, `ranking_version` is
  bumped if the rules changed, and there's a `CHANGELOG.yaml` entry.

> 💡 Can't run the full benchmark yourself (no GPU, dataset too large)? Open a
> [submit-for-me issue](https://github.com/lab260ru/speech_spoof_bench/issues/new?template=submit-for-me.yml)
> (it's labelled `submit-for-me`) and we'll consider running your model for you.

## The thing that surprises everyone

This project is **four loosely-coupled repositories** — the package, the `arena-manifest`,
the Arena Space, and the dataset/model repos — joined only by version pins and commit
SHAs. A change in one is invisible to the others until something is bumped or refreshed,
and most breakage lives in those gaps. When in doubt, read
[architecture/versioning.md](docs/architecture/versioning.md), the authoritative map of
every version number and when you must update it.

Commit and PR style: keep it simple and descriptive; the maintainers will guide anything
project-specific on the PR.
