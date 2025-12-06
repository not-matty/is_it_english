# Is it English?

Goal: given two chunks of English text, decide which one was altered.

## Current Scheme (LM + windowed features + MLP head)

- **LM**: decoder-only byte-level Transformer (`out_lm/ckpt.pt`) trained on original sentences (BOS/EOS).
- **Features**: windowed NLL around the diff span (Δ window NLL, normalized deltas) plus simple length stats; computed with fixed LM weights.
- **Head**: tiny MLP from `model.py` + linear logit trained on labeled pairs via `train_logreg.py` (best ckpt in `out_logreg/ckpt.pt`).
- **Inference**: `inference.py` rebuilds the same features for `data/test.rand.txt` and writes `A`/`B` predictions to `predictions.txt` using the saved head and scaler.
- **Data prep**: `data/prepare.py` / `prepareLM.py` recreate splits, NPZs, and meta with seed 1337.


## Setup

Env:
```
conda env create -f environment.yml
conda activate englishenv
```

Data prep (seed=1337):
```
python data/prepare.py
python data/prepareLM.py
```

LM (already have out_lm/ckpt.pt; retrain if desired):
```
python LM/train.py  # pick hyperparams in the script
```

Logreg head (adjust TRAIN_LIMIT/VAL_LIMIT in train_logreg.py, set RESUME_PATH=None unless resuming):
```
python train_logreg.py
```

Eval:
```
python eval_logreg.py --limit 5000
```

Inference:
```
python inference.py  # reads data/test.rand.txt, writes predictions.txt
```
