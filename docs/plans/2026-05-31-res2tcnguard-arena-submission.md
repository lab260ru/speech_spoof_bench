# Res2TCNGuard Arena Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap Res2TCNGuard as a benchmark model, tune its inference throughput, then run it on all 5 core Arena datasets (one at a time) and open a submission PR per dataset for the user to merge.

**Architecture:** A self-contained wrapper file (network classes copied verbatim from the source notebook + an `AntiSpoofingModel` subclass with deterministic 64600-sample windowing) lives on the dataset drive at `benchmarks/Res2TCNGuard/`. A one-time perf sweep picks `batch_size` and `num_workers` and decides between the canonical runner and a faithful parallel-decode runner. Then a per-dataset loop produces `scores.txt`, uploads it to an HF model repo (commit-pinned), authors a pointer-YAML, and opens a PR on the dataset repo.

**Tech Stack:** Python, PyTorch, `speech-spoof-bench` CLI, Hugging Face Hub (`hf` CLI + `huggingface_hub`), scipy.

**Spec:** `docs/specs/2026-05-31-res2tcnguard-arena-submission-design.md`

**Conventions for every run/CLI step:** prefix GPU work with `CUDA_VISIBLE_DEVICES=3` and `PYTHONPATH=/home/kirill/speech-spoof-bench/benchmarks/Res2TCNGuard`. The source clone is at `/tmp/Res2TCNGuard` (re-clone from `https://github.com/lab260ru/Res2TCNGuard` if gone). Model dir: `/home/kirill/speech-spoof-bench/benchmarks/Res2TCNGuard/` (call it `$MODEL_DIR`). Work/results dir: `$MODEL_DIR/results/`.

---

## File Structure

| Path | Responsibility |
|---|---|
| `$MODEL_DIR/best_1.495.pth` | Pretrained checkpoint (copied from the clone) |
| `$MODEL_DIR/_net.py` | Network classes copied verbatim from `TCN.ipynb` (SincConv_fast, Res2Block, SE_Block, Encoder, Chomp1d, TemporalBlock, TemporalConvNet, TestModel) |
| `$MODEL_DIR/res2tcnguard.py` | `pad_fixed` + `Res2TCNGuard(AntiSpoofingModel)` wrapper; imports from `_net` |
| `$MODEL_DIR/test_res2tcnguard.py` | Unit tests: pad determinism, load, score_batch shape/finiteness |
| `$MODEL_DIR/sweep.py` | One-time batch-size + num_workers throughput sweep |
| `$MODEL_DIR/fast_runner.py` | (Only if §0 decides) faithful parallel-decode runner + score-identity check |
| `$MODEL_DIR/meta.yaml` | Submission system metadata (name/slug/code/checkpoint/paper) |
| `$MODEL_DIR/results/<DATASET>/scores.txt` | Per-dataset score files |
| `$MODEL_DIR/submissions/<DATASET>/Res2TCNGuard.yaml` | Per-dataset pointer-YAML for the PR |
| `implementation-notes.md` (in `$MODEL_DIR`) | Sweep results + decisions log |

---

## Task 1: Set up model dir, checkpoint, and network code

**Files:**
- Create: `$MODEL_DIR/best_1.495.pth`, `$MODEL_DIR/_net.py`

- [ ] **Step 1: Create dir and copy checkpoint**

```bash
MODEL_DIR=/home/kirill/speech-spoof-bench/benchmarks/Res2TCNGuard
mkdir -p "$MODEL_DIR"
test -f /tmp/Res2TCNGuard/best_1.495.pth || git clone --depth 1 https://github.com/lab260ru/Res2TCNGuard /tmp/Res2TCNGuard
cp /tmp/Res2TCNGuard/best_1.495.pth "$MODEL_DIR/best_1.495.pth"
ls -la "$MODEL_DIR/best_1.495.pth"   # expect ~818K
```

- [ ] **Step 2: Extract the network classes verbatim from the notebook into `_net.py`**

The model code is notebook cells 5 (SincConv_fast, Res2Block, SE_Block, Encoder), 6 (Chomp1d, TemporalBlock, TemporalConvNet), and 8 (TestModel). Extract them mechanically (no edits) plus the imports they need:

```bash
python3 - <<'PY'
import json
nb = json.load(open('/tmp/Res2TCNGuard/TCN.ipynb'))
header = (
    "import math\n"
    "import numpy as np\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "from torch.nn.utils import weight_norm\n\n"
)
body = "\n\n".join("".join(nb['cells'][i]['source']) for i in (5, 6, 8))
# drop the duplicate `import torch ...` lines that cell 6 starts with
open('/home/kirill/speech-spoof-bench/benchmarks/Res2TCNGuard/_net.py', 'w').write(header + body + "\n")
print("wrote _net.py")
PY
```

- [ ] **Step 3: Verify `_net.py` imports and instantiates**

```bash
cd "$MODEL_DIR" && python3 -c "import _net; m=_net.TestModel(); print('params_millions=', sum(p.numel() for p in m.parameters())/1e6)"
```
Expected: prints a small number (well under 1.0). Record it for `meta.yaml`.

- [ ] **Step 4: Verify the checkpoint loads cleanly into `TestModel`**

```bash
cd "$MODEL_DIR" && python3 -c "
import torch, _net
m=_net.TestModel()
sd=torch.load('best_1.495.pth', map_location='cpu')
sd=sd.get('state_dict', sd)
missing,unexpected=m.load_state_dict(sd, strict=True)
print('loaded strict OK')
"
```
Expected: `loaded strict OK` with no exception. If keys are prefixed (e.g. `module.`), strip the prefix in this step and note it — the wrapper's `load()` (Task 2) must apply the same stripping.

- [ ] **Step 5: Commit**

```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
# (benchmarks/ is a symlink onto the data drive — not part of this git repo;
#  there is nothing to commit here. Skip if `git status` is clean.)
echo "Task 1 artifacts live on the data drive, not in git."
```

---

## Task 2: Write the wrapper `res2tcnguard.py`

**Files:**
- Create: `$MODEL_DIR/res2tcnguard.py`

- [ ] **Step 1: Write the wrapper**

```python
# res2tcnguard.py
from __future__ import annotations
import os
import numpy as np
import torch
from speech_spoof_bench.model import AntiSpoofingModel
from _net import TestModel

_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_1.495.pth")
_CUT = 64600  # fixed input length the classifier head requires


def pad_fixed(x: np.ndarray, max_len: int = _CUT) -> np.ndarray:
    """Deterministic: first max_len samples; tile-repeat if shorter."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = x.shape[0]
    if n >= max_len:
        return x[:max_len]
    reps = max_len // n + 1
    return np.tile(x, reps)[:max_len].astype(np.float32)


class Res2TCNGuard(AntiSpoofingModel):
    name = "Res2TCNGuard"
    expected_sample_rate = 16000
    batch_size = 32  # overwritten by the §0 sweep result before real runs

    def load(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        net = TestModel()
        sd = torch.load(_CKPT, map_location="cpu")
        sd = sd.get("state_dict", sd)
        net.load_state_dict(sd, strict=True)
        self.net = net.eval().to(self.device)

    @torch.no_grad()
    def score_batch(self, audios: list[np.ndarray], srs: list[int]) -> list[float]:
        x = np.stack([pad_fixed(a) for a in audios])           # (k, 64600)
        xt = torch.from_numpy(x).to(self.device)
        _, logits = self.net(xt)                                # (k, 2)
        return logits[:, 1].detach().cpu().float().tolist()     # higher = bonafide

    def unload(self) -> None:
        self.net = None
```

> If Task 1 Step 4 required stripping a key prefix, apply the identical transform to `sd` in `load()` here.

- [ ] **Step 2: Commit** — n/a (on data drive). Proceed.

---

## Task 3: Unit-test the wrapper

**Files:**
- Create: `$MODEL_DIR/test_res2tcnguard.py`

- [ ] **Step 1: Write the failing tests**

```python
# test_res2tcnguard.py
import numpy as np
import res2tcnguard as R


def test_pad_fixed_long_is_truncated_deterministic():
    x = np.arange(100000, dtype=np.float32)
    out = R.pad_fixed(x)
    assert out.shape == (64600,)
    assert np.array_equal(out, x[:64600])              # first-N, deterministic


def test_pad_fixed_short_is_tiled_to_length():
    x = np.arange(1000, dtype=np.float32)
    out = R.pad_fixed(x)
    assert out.shape == (64600,)
    assert np.array_equal(out[:1000], x)               # starts with original


def test_load_and_score_batch_shape_and_finite():
    m = R.Res2TCNGuard()
    m.load()
    audios = [np.random.randn(48000).astype(np.float32),
              np.random.randn(20000).astype(np.float32)]
    out = m.score_batch(audios, [16000, 16000])
    assert len(out) == 2
    assert all(np.isfinite(s) for s in out)
    # single-item path (runner fallback) must also work
    assert len(m.score_batch(audios[:1], [16000])) == 1
    m.unload()
```

- [ ] **Step 2: Run and verify**

```bash
cd "$MODEL_DIR" && CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$MODEL_DIR" python3 -m pytest test_res2tcnguard.py -v
```
Expected: 3 passed. (Score-*direction* is not unit-testable without labelled audio — it is verified by the ASVspoof2019_LA EER gate in Task 8.)

---

## Task 4: One-time performance sweep (§0)

**Files:**
- Create: `$MODEL_DIR/sweep.py`, append results to `$MODEL_DIR/implementation-notes.md`

- [ ] **Step 1: Write the sweep script**

```python
# sweep.py — batch-size (GPU) + num_workers (decode) throughput sweep
import sys, time, statistics
import numpy as np, torch
import res2tcnguard as R

def time_batch_sizes():
    m = R.Res2TCNGuard(); m.load()
    print("== batch-size sweep (dummy 64600 tensors) ==")
    best = (None, 0.0)
    for bs in [1, 2, 4, 8, 16, 32, 64]:
        audios = [np.random.randn(64600).astype(np.float32) for _ in range(bs)]
        srs = [16000] * bs
        m.score_batch(audios, srs)  # warm-up
        if m.device == "cuda": torch.cuda.synchronize()
        reps, t = 5, []
        for _ in range(reps):
            s = time.perf_counter(); m.score_batch(audios, srs)
            if m.device == "cuda": torch.cuda.synchronize()
            t.append(time.perf_counter() - s)
        ups = bs / statistics.median(t)
        print(f"  bs={bs:>3}  {ups:8.1f} utt/s")
        if ups > best[1]: best = (bs, ups)
    print(f"  -> fastest batch_size={best[0]} ({best[1]:.1f} utt/s)")
    m.unload()
    return best[0]

def time_workers(dataset_id, n=1500):
    from speech_spoof_bench.loader import load_dataset_source  # see note below
    from speech_spoof_bench.runner import _extract
    from torch.utils.data import DataLoader
    print(f"== num_workers sweep (decode {n} rows of {dataset_id}) ==")
    src, ds = load_dataset_source(dataset_id)         # adapt to real loader API
    rows = list(itertools.islice(ds, n))
    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, i):
            utt, arr, sr, lab = _extract(rows[i], 16000)
            return utt, R.pad_fixed(arr), lab
    for nw in [0, 2, 4, 8, 16]:
        dl = DataLoader(DS(), batch_size=32, num_workers=nw,
                        collate_fn=lambda b: b)
        s = time.perf_counter()
        for _ in dl: pass
        dt = time.perf_counter() - s
        print(f"  num_workers={nw:>2}  {n/dt:8.1f} utt/s decode")

if __name__ == "__main__":
    bs = time_batch_sizes()
    # workers sweep is best-effort; adapt loader call to the real API (Task 4 Step 2)
```

> **Note:** the exact loader entry point may differ. Before running, confirm the
> real function in `src/speech_spoof_bench/loader.py` (how the CLI builds a
> `DatasetSource` + iterable for a local-registry dataset) and wire `sweep.py`'s
> `time_workers` to it. The `_extract`/`pad_fixed` calls must stay exactly as the
> wrapper/runner use them.

- [ ] **Step 2: Confirm the loader API, fix `time_workers`, then run the sweep**

```bash
cd "$MODEL_DIR"
grep -n "def load\|DatasetSource\|def iter\|streaming" /home/kirill/speech-spoof-bench/speech-spoof-bench/src/speech_spoof_bench/loader.py | head
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$MODEL_DIR" python3 sweep.py SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
```
Expected: a table of utts/sec per batch size and per num_workers.

- [ ] **Step 3: Record results and decide**

Append to `$MODEL_DIR/implementation-notes.md`: the chosen `batch_size` (fastest), the best `num_workers`, decode utts/sec, and the **runner decision**:
- If parallel decode is **not materially faster** than the canonical runner → use `speech-spoof-bench run` (skip Task 5).
- If it **is** materially faster → build the faithful runner (Task 5).

- [ ] **Step 4: Set the chosen `batch_size` in the wrapper**

Edit `$MODEL_DIR/res2tcnguard.py`: set `batch_size = <fastest from Step 3>`. Re-run Task 3 tests to confirm still green:
```bash
cd "$MODEL_DIR" && CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$MODEL_DIR" python3 -m pytest test_res2tcnguard.py -q
```

---

## Task 5: (Conditional) faithful parallel-decode runner

**Skip this task entirely if Task 4 Step 3 chose the canonical runner.**

**Files:**
- Create: `$MODEL_DIR/fast_runner.py`

- [ ] **Step 1: Write the runner**

It must reuse the package's exact preprocessing (`_extract` / `_to_float32_mono_16k`) and the model's `score_batch`; parallelize only decode via a `DataLoader(num_workers=<best>)`. Write `scores.txt` in the same `utt_id score` format (6 decimals) and apply the same `>5%` skip abort. Model lives in the main process/thread; only decode is workered.

```python
# fast_runner.py
import os, sys, json
import numpy as np, torch
from torch.utils.data import DataLoader, Dataset
import res2tcnguard as R
from speech_spoof_bench.runner import _extract

class _Decode(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        utt, arr, sr, lab = _extract(self.rows[i], 16000)
        return utt, R.pad_fixed(arr), int(lab)

def run(dataset_id, out_path, num_workers):
    from <loader-entrypoint> import <build_iterable>     # from Task 4 Step 2
    src, ds = <build_iterable>(dataset_id)
    rows = list(ds)
    m = R.Res2TCNGuard(); m.load()
    dl = DataLoader(_Decode(rows), batch_size=m.batch_size,
                    num_workers=num_workers, collate_fn=lambda b: b)
    n_total = n_skip = 0
    with open(out_path, "w") as f:
        for batch in dl:
            utts = [b[0] for b in batch]
            arrs = [b[1] for b in batch]
            try:
                scores = m.score_batch(arrs, [16000]*len(arrs))
            except Exception:
                scores = [m.score_batch([a],[16000])[0] for a in arrs]
            for utt, sc in zip(utts, scores):
                n_total += 1
                if sc is None or not np.isfinite(sc): n_skip += 1; continue
                f.write(f"{utt} {sc:.6f}\n")
    m.unload()
    assert n_skip / max(1,n_total) <= 0.05, f"too many skips {n_skip}/{n_total}"
    print(f"wrote {out_path}: {n_total-n_skip} scored, {n_skip} skipped")

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]))
```

- [ ] **Step 2: Score-identity gate (MUST pass before any real use)**

Produce scores with BOTH runners on a subset of ASVspoof2019_LA and assert the per-utt scores match within 1e-6.

```bash
cd "$MODEL_DIR"
# canonical runner, full or capped subset:
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$MODEL_DIR" speech-spoof-bench run \
  --model-module res2tcnguard:Res2TCNGuard \
  --datasets SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA \
  --output-dir ./results_canon
# fast runner:
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$MODEL_DIR" python3 fast_runner.py \
  SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA ./results_fast_scores.txt <best_workers>
python3 - <<'PY'
def load(p):
    d={}
    for ln in open(p):
        u,s=ln.split(); d[u]=float(s)
    return d
a=load("./results_canon/ASVspoof2019_LA/scores.txt")  # adjust slug subdir if needed
b=load("./results_fast_scores.txt")
assert a.keys()==b.keys(), (len(a),len(b))
import math
bad=[(u,a[u],b[u]) for u in a if abs(a[u]-b[u])>1e-6]
print("max diff", max(abs(a[u]-b[u]) for u in a), "mismatches", len(bad))
assert not bad, bad[:5]
print("SCORE-IDENTICAL ✅")
PY
```
Expected: `SCORE-IDENTICAL ✅`. If it fails, do **not** use `fast_runner.py` — fall back to the canonical runner for all datasets.

---

## Task 6: Author `meta.yaml`

**Files:**
- Create: `$MODEL_DIR/meta.yaml`

- [ ] **Step 1: Write meta.yaml**

```yaml
system:
  name: "Res2TCNGuard"
  slug: "Res2TCNGuard"
  description: "TCN-based anti-spoofing system (sinc front-end + Res2Net encoder + dual TCN), ASVspoof2019 LA pretrained, deterministic 64600-sample crop, FP32."
  code: "https://github.com/lab260ru/Res2TCNGuard"
  checkpoint: "https://huggingface.co/SpeechAntiSpoofingBenchmarks/Res2TCNGuard"  # refine to commit-pinned blob URL after Task 7
  params_millions: <value from Task 1 Step 3>
  paper:
    arxiv_id: "10.48084/etasr.8906"
    url: "https://etasr.com/index.php/ETASR/article/view/8906"
    bibtex: |
      @article{Borodin_Kudryavtsev_Mkrtchian_Gorodnichev_2024,
        place={Greece},
        title={Capsule-based and TCN-based Approaches for Spoofing Detection in Voice Biometry},
        volume={14}, number={6},
        url={https://etasr.com/index.php/ETASR/article/view/8906},
        DOI={10.48084/etasr.8906},
        journal={Engineering, Technology & Applied Science Research},
        author={Borodin, Kirill and Kudryavtsev, Vasiliy and Mkrtchian, Grach and Gorodnichev, Mikhail},
        year={2024}, month={Dec.}, pages={18409--18414}
      }
notes: "Deterministic first-64600-sample window (no random crop)."
```

- [ ] **Step 2: Schema-sanity the meta block (offline)**

```bash
cd "$MODEL_DIR" && python3 -c "
import json, yaml, jsonschema
schema=json.load(open('/home/kirill/speech-spoof-bench/speech-spoof-bench/src/speech_spoof_bench/data/submission_meta.schema.json'))
doc=yaml.safe_load(open('meta.yaml'))
jsonschema.validate(doc, schema); print('meta schema OK')
"
```
Expected: `meta schema OK`. (If the schema's top-level requires more than `system`, this surfaces it now.)

---

## Task 7: Create the HF model repo and upload the checkpoint

**Files:** none local (remote HF repo)

- [ ] **Step 1: Create the model repo and upload the checkpoint**

```bash
cd "$MODEL_DIR"
python3 -c "from huggingface_hub import HfApi; HfApi().create_repo('SpeechAntiSpoofingBenchmarks/Res2TCNGuard', repo_type='model', exist_ok=True); print('repo ready')"
hf upload SpeechAntiSpoofingBenchmarks/Res2TCNGuard best_1.495.pth best_1.495.pth --repo-type model
```
Expected: upload succeeds. (Uses the `HF_TOKEN` already in the environment.)

- [ ] **Step 2: Capture the checkpoint commit SHA and finalize `meta.yaml` checkpoint URL**

```bash
python3 -c "from huggingface_hub import HfApi; print(HfApi().model_info('SpeechAntiSpoofingBenchmarks/Res2TCNGuard').sha)"
```
Set `meta.yaml` `system.checkpoint` to the pinned blob URL:
`https://huggingface.co/SpeechAntiSpoofingBenchmarks/Res2TCNGuard/blob/<sha>/best_1.495.pth`.

---

## Task 8: Per-dataset submission loop (5×, sequential — STOP for user merge between each)

Order and manifest-pinned revisions:

| # | DATASET | revision |
|---|---|---|
| 1 | ASVspoof2019_LA | `9492c4a85ad91508b6da03c92c98c58aeaa02424` |
| 2 | ASVspoof2021_DF | `16d4f7d6c68694ac9b0bd43b83df322d1bc5102e` |
| 3 | ASVspoof2021_LA | `dc119733697c946fcd17fe7c1541d7f26b4bbe07` |
| 4 | CD-ADD | `c2de87d49b268b624e6af7440dc2890703098965` |
| 5 | InTheWild | `a957f2582802cdb5964e118818c2e46b3d61aa35` |

Run the steps below for each dataset in order. **After Step 7 for a dataset, STOP and hand the PR link to the user; do not start the next dataset until the user confirms they merged it and checked the Arena.**

- [ ] **Step 0 (InTheWild only, once): register it in the local registry**

```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
speech-spoof-bench local list | grep -q InTheWild || \
  speech-spoof-bench local set SpeechAntiSpoofingBenchmarks/InTheWild \
    /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/InTheWild
speech-spoof-bench local list
```
Expected: InTheWild now listed. (The other 4 are already registered.)

- [ ] **Step 1: Run inference (local copy, GPU 3)**

```bash
cd "$MODEL_DIR"
DS=<DATASET>   # e.g. ASVspoof2019_LA
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$MODEL_DIR" speech-spoof-bench run \
  --model-module res2tcnguard:Res2TCNGuard \
  --datasets SpeechAntiSpoofingBenchmarks/$DS \
  --output-dir ./results
# (or fast_runner.py if Task 5 was adopted)
cat ./results/$DS/result.yaml      # note eer_percent, n_trials, n_skipped
```
Expected: `result.yaml` with sane `n_skipped` (≈0). If a run aborts with `TooManySkips`, debug the wrapper before continuing.

- [ ] **Step 2: SANITY GATE — ASVspoof2019_LA (dataset #1) only**

Confirm `eer_percent` is in the low single digits (near the paper's 1.49%; deterministic crop may shift it a little). **If EER ≈ 98% the score direction is flipped** — fix `score_batch` (return `-logits[:,1]` or check class index) and re-run before proceeding. Out-of-domain datasets (#2–#5) are expected to show high EER; that is not a failure.

- [ ] **Step 3: Upload `scores.txt` to the model repo (commit-pinned)**

```bash
cd "$MODEL_DIR"
hf upload SpeechAntiSpoofingBenchmarks/Res2TCNGuard \
  ./results/$DS/scores.txt \
  .eval_results/SpeechAntiSpoofingBenchmarks/$DS/scores.txt \
  --repo-type model
SHA=$(python3 -c "from huggingface_hub import HfApi; print(HfApi().model_info('SpeechAntiSpoofingBenchmarks/Res2TCNGuard').sha)")
SCORES_SHA256=$(sha256sum ./results/$DS/scores.txt | cut -d' ' -f1)
echo "model_sha=$SHA scores_sha256=$SCORES_SHA256"
```

- [ ] **Step 4: Author the submission pointer-YAML**

```bash
mkdir -p "$MODEL_DIR/submissions/$DS"
cp /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/$DS/submissions/results_template.yaml \
   "$MODEL_DIR/submissions/$DS/Res2TCNGuard.yaml"
```
Then edit `$MODEL_DIR/submissions/$DS/Res2TCNGuard.yaml`:
- `system:` block ← copy from `meta.yaml` (including the finalized commit-pinned `checkpoint` URL and `params_millions`).
- `dataset.id: SpeechAntiSpoofingBenchmarks/$DS`, `dataset.revision: <manifest-pinned revision from the table>`, `dataset.split: test`.
- `scores.eer_percent / n_trials / n_skipped` ← from `result.yaml`.
- `artifact.scores_url: https://huggingface.co/SpeechAntiSpoofingBenchmarks/Res2TCNGuard/resolve/<model_sha>/.eval_results/SpeechAntiSpoofingBenchmarks/$DS/scores.txt`
- `artifact.scores_sha256: <SCORES_SHA256>`
- `artifact.bench_version:` ← `speech-spoof-bench=={version}` (from `python3 -c "import speech_spoof_bench,importlib.metadata as m; print(m.version('speech-spoof-bench'))"`).
- `submitter.hf_username: SpeechAntiSpoofingBenchmarks`, `submitter.contact: k.n.borodin@mtuci.ru`, `submitted_at: 2026-05-31`.
- Leave `reproduction:` empty (maintainer fills at merge).

- [ ] **Step 5: Validate the submission YAML offline (schema only — NO `reproduce`)**

```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
speech-spoof-bench validate-submission "$MODEL_DIR/submissions/$DS/Res2TCNGuard.yaml"
```
Expected: validation passes. Fix any schema errors before opening the PR.

- [ ] **Step 6: Open the PR on the dataset repo**

```bash
cd "$MODEL_DIR/submissions/$DS"
hf upload SpeechAntiSpoofingBenchmarks/$DS \
  Res2TCNGuard.yaml submissions/Res2TCNGuard.yaml \
  --repo-type dataset --create-pr
```
Expected: prints a PR/discussion URL.

- [ ] **Step 7: Hand off to the user and STOP**

Report to the user: dataset, EER, and the PR URL. Ask them to **merge the PR and check the Arena**, then confirm before you start the next dataset. Do not proceed automatically.

---

## Self-Review notes

- **Spec coverage:** §0 sweep → Task 4(+5); wrapper/windowing/score-direction → Tasks 2–3 + Task 8 Step 2 gate; model repo → Task 7; meta.yaml/paper Path A → Task 6; per-dataset loop + pinned revisions + local source + no-CI-mirror + user-merge handoff → Task 8; InTheWild registration → Task 8 Step 0; model location on benchmarks drive → Task 1.
- **Skipped by design (per user):** `reproduce --scoring --no-local` CI mirror is intentionally omitted.
- **Open API detail:** the dataset-loader entry point used by `sweep.py`/`fast_runner.py` (Task 4 Step 2 / Task 5) must be confirmed against `loader.py` at execution time — flagged inline, not a silent assumption.
