import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from LM.model import GPT, GPTConfig, MLP

# Paths
LOGREG_CKPT = os.path.join("out_logreg", "ckpt.pt")
PAIRS_PATH = os.path.join("data", "test.rand.txt")
META_PATH = os.path.join("data", "train_lm", "meta.pkl")
OUT_PATH = "part1.txt"


def diff_span(x: bytes, y: bytes) -> Tuple[int | None, int | None]:
    """Return (start, end) index of the differing span, or (None, None) if equal."""
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


def read_pairs(path: str) -> List[Tuple[bytes, bytes]]:
    pairs = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(b"\t")
            if len(parts) != 2:
                continue
            pairs.append((parts[0], parts[1]))
    return pairs


def load_lm(ckpt_path: str, device: str) -> GPT:
    ckpt = torch.load(ckpt_path, map_location=device)
    margs = ckpt["model_args"]
    model = GPT(GPTConfig(**margs))
    sd = ckpt["model"]
    for k in list(sd):
        if k.startswith("_orig_mod."):
            sd[k[10:]] = sd.pop(k)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def load_meta():
    meta = pickle.load(open(META_PATH, "rb"))
    return meta["bos_id"], meta["eos_id"]


def seq_nll(model: GPT, b: bytes, BOS: int, EOS: int, block_size: int, device: str) -> float:
    """Sum NLL over the sequence using chunked forward passes."""
    x = np.concatenate(([BOS], np.frombuffer(b, dtype=np.uint8), [EOS])).astype(np.int64)
    nll = 0.0
    for i in range(0, len(x), block_size):
        chunk = torch.from_numpy(x[i : i + block_size])
        if len(chunk) < 2:
            break
        X = chunk[:-1].unsqueeze(0).to(device)
        Y = chunk[1:].unsqueeze(0).to(device)
        with torch.no_grad():
            _, loss = model(X, Y)
        nll += loss.item() * Y.numel()
    return nll


def build_features(model: GPT, pairs, BOS: int, EOS: int, block_size: int, window_max_len: int, device: str):
    feats = []
    for A, B in pairs:
        lenA, lenB = len(A), len(B)
        wa, wb = trim_around_diff(A, B, min(window_max_len, block_size - 2))
        wnA = seq_nll(model, wa, BOS, EOS, block_size, device)
        wnB = seq_nll(model, wb, BOS, EOS, block_size, device)

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
    feats = torch.tensor(feats, dtype=torch.float32)
    return feats


def main():
    if not os.path.exists(LOGREG_CKPT):
        raise FileNotFoundError(f"Logreg checkpoint not found at {LOGREG_CKPT}")
    logreg = torch.load(LOGREG_CKPT, map_location="cpu")
    lm_ckpt = logreg["config"].get("ckpt_path", os.path.join("out_lm", "ckpt.pt"))
    window_max_len = logreg["config"].get("window_max_len", 256)
    mean = logreg["mean"]
    std = logreg["std"]
    feature_dim = logreg["feature_dim"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading LM from {lm_ckpt} on {device} ...")
    lm = load_lm(lm_ckpt, device)
    BOS, EOS = load_meta()
    block_size = lm.config.block_size

    pairs = read_pairs(PAIRS_PATH)
    print(f"Scoring {len(pairs)} pairs from {PAIRS_PATH}")

    feats = build_features(lm, pairs, BOS, EOS, block_size, window_max_len, device)
    feats_std = (feats - mean) / std

    cfg = type("Cfg", (), {"n_embd": feature_dim, "bias": True, "dropout": 0.0})
    head = nn.Sequential(MLP(cfg), nn.Linear(feature_dim, 1))
    head.load_state_dict(logreg["head_state_dict"])
    head.to(device).eval()

    with torch.no_grad():
        logits = head(feats_std.to(device)).squeeze(1).cpu()
    preds = ["A" if logit.item() > 0 else "B" for logit in logits]

    with open(OUT_PATH, "w") as f:
        for p in preds:
            f.write(p + "\n")
    print(f"Wrote predictions to {OUT_PATH}")


if __name__ == "__main__":
    main()
