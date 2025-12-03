# Is it English?

Goal: given two chunks of English text, decide which one was altered.

## Current Scheme (LM + windowed features + MLP head)

- **LM**: decoder-only byte-level Transformer (`out_lm/ckpt.pt`) trained on original sentences (BOS/EOS).
- **Features**: windowed NLL around the diff span (Δ window NLL, normalized deltas) plus simple length stats; computed with fixed LM weights.
- **Head**: tiny MLP from `model.py` + linear logit trained on labeled pairs via `train_logreg.py` (best ckpt in `out_logreg/ckpt.pt`).
- **Inference**: `inference.py` rebuilds the same features for `data/test.rand.txt` and writes `A`/`B` predictions to `predictions.txt` using the saved head and scaler.
- **Data prep**: `data/prepare.py` / `prepareLM.py` recreate splits, NPZs, and meta with seed 1337.


## Todo

- **bidirectional (no causal mask)**  
  Concatenate `A [SEP] B` and train a small cross-encoder with a logistic loss to choose the altered one. Or just use an encoder-style Transformer and classify directly. (removing causal mask)

- **discriminator on top of the LM**  
  Keep the current LM and add a tiny head (e.g., logistic regression or a small MLP) over simple features like `ΔNLL`, length terms, maybe pooled hidden states.

- **test larger vocab**  
  Train a tokenizer (e.g., BPE) and compare against bytes. 
