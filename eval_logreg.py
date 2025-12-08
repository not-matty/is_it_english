"""
Evaluate the saved logistic-regression head on validation pairs
load LM + head, build features, and report accuracy. Should try pooling hidden state if necessary for better accuracy?

Example:
$ python eval_logreg.py --ckpt out_logreg/ckpt.pt --val data/train/val_pairs.tsv --limit 10000
"""

import argparse
import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from LM.model import GPT, GPTConfig, MLP


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


def load_lm(ckpt_path: str, device: str) -> GPT:
    # mirror LM/train.py loading; strip any _orig_mod prefix that torch.compile may add
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


def load_meta(meta_path: str):
    meta = pickle.load(open(meta_path, "rb"))
    return meta["bos_id"], meta["eos_id"]


def seq_nll(model: GPT, b: bytes, BOS: int, EOS: int, block_size: int, device: str) -> float:
    """Sum NLL over the sequence using chunked forward passes."""
    x = np.concatenate(([BOS], np.frombuffer(b, dtype=np.uint8), [EOS])).astype(np.int64)
    nll = 0.0
    # process in chunks wrt block_size
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


def build_features(model: GPT, pairs, BOS: int, EOS: int, block_size: int, window_max_len: int, device: str):
    # compute feature vector per pair based on LM NLL deltas and lengths
    feats = []
    labels = []
    for label, A, B in pairs:
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
        labels.append(label)
    feats = torch.tensor(feats, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    return feats, labels


def main():
    ap = argparse.ArgumentParser(description="Evaluate logreg head on validation pairs.")
    ap.add_argument("--ckpt", default=os.path.join("out_logreg", "ckpt.pt"), help="Path to saved logreg checkpoint.")
    ap.add_argument("--val", default=os.path.join("data", "train", "val_pairs.tsv"), help="Validation TSV path.")
    ap.add_argument("--limit", type=int, default=100000, help="Number of val samples to evaluate (None for all).")
    ap.add_argument("--meta", default=os.path.join("data", "train_lm", "meta.pkl"), help="Path to meta.pkl with BOS/EOS.")
    ap.add_argument("--lm", default=None, help="Override LM checkpoint path (defaults to value in logreg ckpt).")
    ap.add_argument("--window", type=int, default=256, help="Diff-centered window length.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")
    lm_ckpt_path = args.lm or ckpt["config"].get("ckpt_path")
    if lm_ckpt_path is None:
        raise ValueError("LM checkpoint path not found; provide --lm.")

    print(f"Loading LM from {lm_ckpt_path} on {device} ...")
    lm = load_lm(lm_ckpt_path, device)
    BOS, EOS = load_meta(args.meta)
    block_size = lm.config.block_size

    print("Loading validation pairs ...")
    val_pairs = load_pairs(args.val, args.limit)
    print(f"Evaluating on {len(val_pairs)} pairs")

    feats, labels = build_features(lm, val_pairs, BOS, EOS, block_size, args.window, device)
    mean = ckpt["mean"]
    std = ckpt["std"]
    feats_std = (feats - mean) / std  # standardize using training-time stats

    cfg = type("Cfg", (), {"n_embd": feats.shape[1], "bias": True, "dropout": 0.0})
    head = nn.Sequential(MLP(cfg), nn.Linear(feats.shape[1], 1))
    head.load_state_dict(ckpt["head_state_dict"])
    head.to(device).eval()

    with torch.no_grad():
        logits = head(feats_std.to(device)).squeeze(1)
        preds = (logits > 0).float().cpu()
    acc = (preds == labels).float().mean().item()
    print(f"Validation accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
