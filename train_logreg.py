"""
Train a lightweight logistic regression head on top of LM NLL-based features.
Features include:
- full-sentence NLLs for A and B and their deltas/normalized deltas
- windowed NLLs (trimmed around the diff span)
- simple length features
The LM weights stay frozen; only the linear head is trained.
"""

import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
import wandb

from LM.model import GPT, GPTConfig, MLP

# Default to offline to avoid network issues unless explicitly overridden.
os.environ.setdefault("WANDB_MODE", "online")

# --- config ---
CKPT_PATH = os.path.join("out_lm", "ckpt.pt")
TRAIN_PATH = os.path.join("data", "train", "train_pairs.tsv")
VAL_PATH = os.path.join("data", "train", "val_pairs.tsv")
META_PATH = os.path.join("data", "train_lm", "meta.pkl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = os.path.join("out_logreg", "ckpt.pt")
RESUME_PATH = None

# Limits to keep runtime manageable; set to None to use all pairs.
TRAIN_LIMIT = 100000
VAL_LIMIT = 5000

# How wide a diff-centered window to score (before adding BOS/EOS).
WINDOW_MAX_LEN = 256

BATCH_SIZE = 512
EPOCHS = 100
LR = 0.05


def diff_span(x: bytes, y: bytes):
    """Return (start, end) index of the first/last differing byte, or (None, None) if equal."""
    minlen = min(len(x), len(y))
    start = None
    for i in range(minlen):
        if x[i] != y[i]:
            start = i
            break
    if start is None:
        if len(x) != len(y):
            start = minlen
        else:
            return None, None
    end = minlen - 1
    while end >= 0 and x[end] == y[end]:
        end -= 1
    if end < start:
        end = start
    return start, end


def trim_around_diff(A: bytes, B: bytes, max_len: int) -> Tuple[bytes, bytes]:
    """Center a window around the diff span so both A/B are aligned and trimmed."""
    ds, de = diff_span(A, B)
    if ds is None:
        return A[:max_len], B[:max_len]
    diff_len = de - ds + 1
    budget = max_len - diff_len
    left_budget = budget // 2
    right_budget = budget - left_budget
    start_a = max(0, ds - left_budget)
    end_a = min(len(A), de + 1 + right_budget)
    trimmed_a = A[start_a:end_a]
    start_b = start_a
    end_b = end_a
    trimmed_b = B[start_b:end_b]
    if len(trimmed_a) > max_len:
        trimmed_a = trimmed_a[:max_len]
    if len(trimmed_b) > max_len:
        trimmed_b = trimmed_b[:max_len]
    return trimmed_a, trimmed_b


def load_model() -> GPT:
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    margs = ckpt["model_args"]
    model = GPT(GPTConfig(**margs))
    sd = ckpt["model"]
    for k in list(sd):
        if k.startswith("_orig_mod."):
            sd[k[10:]] = sd.pop(k)
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    return model


def load_meta():
    meta = pickle.load(open(META_PATH, "rb"))
    BOS, EOS = meta["bos_id"], meta["eos_id"]
    return BOS, EOS


def seq_nll(model: GPT, b: bytes, BOS: int, EOS: int, block_size: int) -> float:
    """Sum NLL over the sequence using chunked forward passes."""
    x = np.concatenate(([BOS], np.frombuffer(b, dtype=np.uint8), [EOS])).astype(np.int64)
    nll = 0.0
    for i in range(0, len(x), block_size):
        chunk = torch.from_numpy(x[i : i + block_size])
        if len(chunk) < 2:
            break
        X = chunk[:-1].unsqueeze(0).to(DEVICE)
        Y = chunk[1:].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, loss = model(X, Y)
        nll += loss.item() * Y.numel()
    return nll


def load_pairs(path: str, limit: int | None) -> List[Tuple[int, bytes, bytes]]:
    pairs = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(b"\t")
            if len(parts) != 3:
                continue
            label = int(parts[0])
            A, B = parts[1], parts[2]
            pairs.append((label, A, B))
            if limit is not None and len(pairs) >= limit:
                break
    return pairs


def build_features(model: GPT, pairs, BOS: int, EOS: int, block_size: int, window_max_len: int):
    feats = []
    labels = []
    for label, A, B in pairs:
        lenA, lenB = len(A), len(B)
        wa, wb = trim_around_diff(A, B, min(window_max_len, block_size - 2))
        wnA = seq_nll(model, wa, BOS, EOS, block_size)
        wnB = seq_nll(model, wb, BOS, EOS, block_size)

        delta_window = wnB - wnA

        tokenA = max(lenA, 1)
        tokenB = max(lenB, 1)
        delta_window_avg = (wnB / max(len(wb), 1)) - (wnA / max(len(wa), 1))
        total_len = max(lenA + lenB, 1)

        feat = [
            delta_window,
            delta_window / total_len,
            delta_window_avg,
            lenA,
            lenB,
            lenA - lenB,
            abs(lenA - lenB),
            (lenA / tokenB) if tokenB else 0.0,
        ]
        feats.append(feat)
        labels.append(label)
    feats = torch.tensor(feats, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    return feats, labels


def standardize(train_x: torch.Tensor, x: torch.Tensor):
    mean = train_x.mean(0, keepdim=True)
    std = train_x.std(0, unbiased=False, keepdim=True).clamp(min=1e-6)
    return (x - mean) / std, mean, std


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    with torch.no_grad():
        logits = model(x).squeeze(1)
        preds = (logits > 0).float()
        acc = (preds == y).float().mean().item()
    return acc


def main():
    print(f"Loading LM from {CKPT_PATH} on {DEVICE} ...")
    lm = load_model()
    BOS, EOS = load_meta()
    block_size = lm.config.block_size
    print(f"LM block size: {block_size}")

    print("Loading pairs ...")
    train_pairs = load_pairs(TRAIN_PATH, TRAIN_LIMIT)
    val_pairs = load_pairs(VAL_PATH, VAL_LIMIT)
    print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

    print("Building features")
    train_x, train_y = build_features(lm, train_pairs, BOS, EOS, block_size, WINDOW_MAX_LEN)
    val_x, val_y = build_features(lm, val_pairs, BOS, EOS, block_size, WINDOW_MAX_LEN)

    resume_ckpt = torch.load(RESUME_PATH, map_location="cpu") if RESUME_PATH and os.path.exists(RESUME_PATH) else None
    if resume_ckpt is not None:
        mean = resume_ckpt["mean"]
        std = resume_ckpt["std"]
        print(f"Resuming scaler from {RESUME_PATH}")
        train_x_std = (train_x - mean) / std
        val_x_std = (val_x - mean) / std
    else:
        train_x_std, mean, std = standardize(train_x, train_x)
        val_x_std = (val_x - mean) / std
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    run = wandb.init(
        project="is_it_english_logreg",
        config=dict(
            ckpt_path=CKPT_PATH,
            train_path=TRAIN_PATH,
            val_path=VAL_PATH,
            train_limit=TRAIN_LIMIT,
            val_limit=VAL_LIMIT,
            window_max_len=WINDOW_MAX_LEN,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            lr=LR,
            device=DEVICE,
            block_size=block_size,
            resume_path=RESUME_PATH,
            head="mlp+linear",
        ),
    )

    # MLP head from model.py followed by final logit
    cfg = type("Cfg", (), {"n_embd": train_x.shape[1], "bias": True, "dropout": 0.0})
    head = nn.Sequential(MLP(cfg), nn.Linear(train_x.shape[1], 1)).to(DEVICE)
    if resume_ckpt is not None:
        head.load_state_dict(resume_ckpt["head_state_dict"])
        print(f"Resumed head from {RESUME_PATH}")
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    train_ds = torch.utils.data.TensorDataset(train_x_std.to(DEVICE), train_y.to(DEVICE))
    val_ds = torch.utils.data.TensorDataset(val_x_std.to(DEVICE), val_y.to(DEVICE))

    def batches(ds):
        loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        for xb, yb in loader:
            yield xb, yb

    best_val = -1.0
    best_epoch = None

    for epoch in range(1, EPOCHS + 1):
        head.train()
        total_loss = 0.0
        for xb, yb in batches(train_ds):
            opt.zero_grad()
            logits = head(xb).squeeze(1)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        avg_loss = total_loss / len(train_ds)
        head.eval()
        train_acc = evaluate(head, train_x_std.to(DEVICE), train_y.to(DEVICE))
        val_acc = evaluate(head, val_x_std.to(DEVICE), val_y.to(DEVICE))
        print(f"Epoch {epoch}: loss={avg_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        wandb.log(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "head_state_dict": head.state_dict(),
                    "mean": mean.cpu(),
                    "std": std.cpu(),
                    "feature_dim": train_x.shape[1],
                    "config": {
                        "ckpt_path": CKPT_PATH,
                        "train_limit": TRAIN_LIMIT,
                        "val_limit": VAL_LIMIT,
                        "window_max_len": WINDOW_MAX_LEN,
                        "batch_size": BATCH_SIZE,
                        "epochs": EPOCHS,
                        "lr": LR,
                        "block_size": block_size,
                        "device": DEVICE,
                    },
                },
                SAVE_PATH,
            )
            print(f"Saved new best checkpoint (val_acc={best_val:.4f}) to {SAVE_PATH}")

    print("Done. You can reuse mean/std and head weights for inference.")
    if best_epoch is not None:
        print(f"Best val_acc={best_val:.4f} at epoch {best_epoch}; checkpoint at {SAVE_PATH}")
    run.finish()


if __name__ == "__main__":
    main()
