# Is it English?

Goal: given two chunks of English text, decide which one was altered.

## Current Scheme (LM + windowed features + MLP head)

**LM**: decoder-only byte-level Transformer (`out_lm/ckpt.pt`) trained on original sentences (BOS/EOS). Training and model architecture from nanoGPT, with implementation of RMSNorm.

**Features**: windowed NLL around the diff span (Δ window NLL, normalized deltas) plus simple length features computed with fixed LM weights.

**Head**: MLP block from `model.py` and linear logit trained on ~100k labeled pairs via `train_logreg.py` with BCEWithLogitsLoss.

**Inference**: `inference.py` rebuilds the same features for `data/test.rand.txt` and writes `A`/`B` predictions to `predictions.txt` using the saved head and scaler.

**Data prep**: `eval_logreg.py` tests classifier on 100k validation set, with around ~97% accuracy.

**Ruin English**: `ruin_english.ipynb` makes simple alterations found in train.txt, but also "cut", where the original sentence is cut down somewhere with a space/punctuation and orders are reversed (like a cut to a deck of cards), and change last word with the trained LM, which should in theory lower nll with temperature 0 sampling. 


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
