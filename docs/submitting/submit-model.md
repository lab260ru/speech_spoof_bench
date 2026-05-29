# Submit a model — step-by-step

New here? This guide assumes no prior knowledge of the toolkit. By the end you'll have
your model's result on the Arena leaderboard.

> 💡 **Can't run the full benchmark yourself?** (No GPU, dataset too big, etc.) If you
> can run your model over the complete dataset(s), email
> **[k.n.borodin@mtuci.ru](mailto:k.n.borodin@mtuci.ru)** and we'll consider running it
> for you.

## What you're actually doing

A speech anti-spoofing model listens to an audio clip and decides whether it's **real
human speech (`bonafide`)** or **synthetic/replayed (`spoof`)**. To get on the
leaderboard you:

1. write a tiny adapter so the toolkit can call your model,
2. let the toolkit score every clip in a dataset → produces a `scores.txt`,
3. put that `scores.txt` in your own Hugging Face (HF) repo,
4. open a pull request (PR) that points at it,
5. a maintainer re-checks the numbers and merges → you appear on the Arena.

Two terms you'll see:

- **EER (Equal Error Rate), %** — the headline metric. **Lower is better** (0 = perfect,
  ~50 = random guessing).
- **score** — your model outputs one number per clip. **Convention: a higher score means
  "more likely bonafide."** If your model outputs a "spoof probability," negate it.

## Prerequisites

- **Python 3.10+.** Check with `python --version`.
- **A Hugging Face account** — sign up free at https://huggingface.co/join.
- **The HF CLI, logged in:**
  ```bash
  pip install huggingface_hub
  huggingface-cli login        # paste a token from huggingface.co/settings/tokens
  ```
- **A model repo you own** to hold your scores. Create one at
  https://huggingface.co/new (type: *Model*), e.g. `your-username/my-model`.

## Step 1 — Wrap your model

The toolkit calls your model through a small class. Create `my_model.py`:

```python
from speech_spoof_bench.model import AntiSpoofingModel
import numpy as np

class MyModel(AntiSpoofingModel):
    name = "my-model"
    expected_sample_rate = 16000          # audio is given to you at 16 kHz mono

    def load(self):
        # Runs ONCE before any scoring. Load weights here.
        self.net = load_my_checkpoint("path/or/hf/repo")

    def score(self, audio: np.ndarray, sr: int) -> float:
        # audio: 1-D float32 numpy array, sr == 16000.
        # Return ONE number; higher == more likely bonafide.
        return float(self.net(audio))

    def unload(self):
        # Runs ONCE after scoring. Free GPU memory etc.
        del self.net
```

That's the minimum. If your model is faster in batches, also implement
`score_batch(self, audios, srs) -> list[float]` (otherwise it just calls `score` in a
loop). If a single clip errors it's skipped; if more than 5% are skipped the dataset run
aborts so you notice.

## Step 2 — Install the toolkit

```bash
pip install speech-spoof-bench
```

## Step 3 — Run the benchmark

```bash
speech-spoof-bench run --model-module my_model:MyModel --datasets all
```

- `--model-module` is `file_or_import_path:ClassName`.
- `--datasets all` runs every Core dataset; or pass one id, e.g.
  `--datasets SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA`.

The toolkit streams the audio (you don't pre-download the whole dataset), calls your
model, and writes per dataset:

```
results/<dataset>/scores.txt     # lines of:  <utterance_id> <score>
results/<dataset>/result.yaml     # the computed EER and metadata
```

Sanity check: a random model lands near **EER 50%**. A real model should be much lower.

## Step 4 — Upload your scores to your model repo

Your scores live in *your* repo (not the dataset repo), at this **exact** path:

```
.eval_results/<dataset-org>/<dataset-name>/scores.txt
```

For ASVspoof2019_LA that's
`.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt`. Upload it:

```bash
huggingface-cli upload your-username/my-model \
  results/ASVspoof2019_LA/scores.txt \
  .eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt
```

(Why your repo? So the leaderboard can link to immutable, sha-pinned scores without
storing big files itself.)

## Step 5 — Describe your system: `meta.yaml`

Create `meta.yaml`. Every field below matters; `params_millions` is optional but powers
the **By model size** tab.

```yaml
system:
  name: My Model                 # shown on the board
  slug: my-model                 # lowercase + hyphens, unique; used in URLs
  description: One line — architecture, config, precision (e.g. "AASIST, FP32").
  code: https://github.com/you/my-model          # source code
  checkpoint: https://huggingface.co/you/my-model # weights
  params_millions: 52.3          # optional: parameter count in millions
  paper:
    arxiv_id: "2110.01200"
    url: https://arxiv.org/abs/2110.01200
    bibtex: |
      @inproceedings{your2022paper,
        title = {...}, author = {...}, year = {2022}
      }
```

No paper yet? Point `paper` at the most relevant reference (e.g. the dataset paper) and
say so in `description`.

> **Paper policy.** A paper is what lets your model compete in the ranked tiers
> (🥇 Gold / 🥈 Silver / 🥉 Bronze). **Proprietary or unpublished models may omit
> `paper` entirely** — they're still welcome, but they appear in the dedicated
> **🔓 Unpublished / Proprietary** tier and are left *unranked*, no matter how strong
> their scores. Add a paper later to move into the ranked tiers.

## Step 6 — Open the submission PR

One command does the rest — re-runs, uploads scores, writes the submission file, and
opens a pull request on the dataset repo:

```bash
speech-spoof-bench submit \
  --model-module my_model:MyModel --datasets all \
  --model-repo your-username/my-model --submission-meta meta.yaml \
  --hf-username your-username --contact you@example.com
```

(A *pull request* is a proposed change a maintainer reviews. You never edit the dataset
repo directly.)

## Step 7 — What happens next (verification)

A maintainer runs a quick check (`reproduce --scoring`): it downloads your `scores.txt`,
verifies it's unchanged, and recomputes the EER — it must match your claim. If it does,
your row is merged with a **✔ scoring** badge. They may also re-run your model
end-to-end, which upgrades you to **★ inference**. This is usually quick but can take
longer for the full inference re-run.

## Step 8 — Add the badge to your model

After merge, an automated comment gives you a ready-to-paste badge snippet. Add it to
your model's README so visitors can jump to your Arena row.

## Common mistakes

- **Scores inverted** — remember *higher = bonafide*. If your EER is ~ (100 − expected),
  you've got the sign backwards.
- **Wrong upload path** — the `.eval_results/<org>/<name>/scores.txt` path must match the
  dataset id exactly.
- **`slug` collisions / capitals** — keep it lowercase-hyphenated and unique.

---

Stuck on any step? Email **[k.n.borodin@mtuci.ru](mailto:k.n.borodin@mtuci.ru)** — happy
to help.
