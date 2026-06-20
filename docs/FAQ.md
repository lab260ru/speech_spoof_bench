# FAQ — the things that silently trip people up

Three failures cause most of the confused emails. Each one passes locally and looks like
something else. Here's how to recognise them.

## 1. My EER is huge (like 97%) — is my model that bad?

Almost certainly not — your **score sign is flipped**. The convention is **higher score =
more likely bonafide** (real human speech), and **label 0 = bonafide, 1 = spoof**. If your
model outputs a *spoof* probability `p`, return `-p` (or `1 - p`) instead. A backwards sign
gives an EER of roughly `100 − the true value` (e.g. 97% instead of 3%) — a dead giveaway.

→ Full detail in [developing/new-model.md](developing/new-model.md#the-three-things-that-bite-everyone).

## 2. `ModuleNotFoundError: No module named 'my_model'`

Python can't find your model file because it only looks in its own folders, not the one
you're standing in. Tell it to also look here by adding `PYTHONPATH=.` in front of the
command (the `.` means "this folder"):

```bash
PYTHONPATH=. speech-spoof-bench run --model-module my_model:MyModel ...
```

→ Full detail in [developing/new-model.md](developing/new-model.md#step-2--run-it-locally-offline).

## 3. My PR merged but my model isn't on the leaderboard

It's not instant — wait one refresh cycle. The Arena re-reads everything on a timer
(about **30 minutes**, or 60 seconds after a failed refresh) plus the automatic refresh
that fires when your PR merges. Give it a refresh cycle before concluding the merge
"didn't work"; you can also press the **🔄 Refresh** button on the Space.

→ Full detail in [developing/testing-and-pitfalls.md](developing/testing-and-pitfalls.md#the-pin--refresh-gaps-the-big-ones).

## 4. Why is my model unranked even though its scores are great?

A **paper** is what lets a model compete in the ranked tiers (🥇 Gold / 🥈 Silver /
🥉 Bronze). Models with no paper are welcome, but they sit in the **🔓 Unpublished /
Proprietary** tier and are left unranked — no matter how strong their scores. Add a paper
later to move into the ranked tiers.

→ Full detail in [submitting/submit-model.md](submitting/submit-model.md#step-5--describe-your-system-metayaml)
and [submitting/how-ranking-works.md](submitting/how-ranking-works.md).
