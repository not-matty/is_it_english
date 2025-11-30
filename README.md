# Altered Sentence Prediction

Goal: given two chunks of English text, decide which one was altered.

## Current Implementation (cross-encoder)

- **Model**: bidirectional byte-level Transformer encoder (RMSNorm, SiLU MLPs) with CLS pooling and an MLP classifier head.
- **Inputs**: `[CLS] A [SEP] B [SEP]` bytes, with token type ids (0 for A, 1 for B), PAD/MASK tokens; block size from `meta.pkl`.
- **Training**: two stages supported: (1) optional MLM pretrain on original sentences only (`train_mlm.npz`), (2) classification finetune on randomized A/B pairs (`train_ce.npz`) with BCE loss.
- **Data prep**: `data/prepare.py` builds cross-encoder NPZs (`train_ce.npz`/`val_ce.npz`) and MLM NPZs (`train_mlm.npz`/`val_mlm.npz`), plus TSVs for inspection.
- **Evaluation rule**: classifier logits directly predict whether the first segment is original.

## Previous LM Baseline

- **Model**: decoder-only, byte-level Transformer (~62M params), trained from scratch (nanoGPT).
- **Training**: causal mask, trained only on **original** sentences.
- **Decision rule**: score each candidate by **summed NLL** under the LM; the one with **lower loss** is treated as the original.
- **Scale**: ~900k pairs (typically 1–2 sentences).
- **Result**: **81.6%** accuracy.

## How the LM baseline worked

Train a byte LM with BOS/EOS. At eval, split long strings into 128-token windows, sum token NLLs for A and B, pick the smaller sum.


## Todo

- **bidirectional (no causal mask)**  
  Concatenate `A [SEP] B` and train a small cross-encoder with a logistic loss to choose the altered one. Or just use an encoder-style Transformer and classify directly. (removing causal mask)

- **discriminator on top of the LM**  
  Keep the current LM and add a tiny head (e.g., logistic regression or a small MLP) over simple features like `ΔNLL`, length terms, maybe pooled hidden states.

- **test larger vocab**  
  Train a tokenizer (e.g., BPE) and compare against bytes. 



![val loss](val_loss.png "val loss")
