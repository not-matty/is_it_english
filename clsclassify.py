import argparse, os, pickle, torch
import numpy as np
from model import GPT, GPTConfig

def trim_around_diff(A: bytes, B: bytes, block_size: int):
    max_len = block_size - 3  # reserve CLS and two SEP
    def diff_span(x: bytes, y: bytes):
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

@torch.no_grad()
def encode_pair(A: bytes, B: bytes, cls_id, sep_id, pad_id, block_size):
    ta, tb = trim_around_diff(A, B, block_size)
    ids = [cls_id] + list(ta) + [sep_id] + list(tb) + [sep_id]
    ids = ids[:block_size]
    types = [0] * (1 + len(ta) + 1) + [1] * (min(len(tb), block_size - len(ids) + len(tb)) + 1)
    types = types[:len(ids)]
    attn = [1] * len(ids)
    if len(ids) < block_size:
        pad_len = block_size - len(ids)
        ids += [pad_id] * pad_len
        types += [1] * pad_len
        attn += [0] * pad_len
    return (
        torch.tensor([ids], dtype=torch.long),
        torch.tensor([attn], dtype=torch.long),
        torch.tensor([types], dtype=torch.long),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="path to ckpt.pt")
    ap.add_argument("--pairs", default=os.path.join("data", "train", "val_pairs.tsv"))
    ap.add_argument("--meta", default=os.path.join("data", "train", "meta.pkl"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    meta = pickle.load(open(args.meta, "rb"))
    cls_id, sep_id, pad_id, block_size = meta["cls_id"], meta["sep_id"], meta["pad_id"], meta["block_size"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    margs = ckpt["model_args"]
    model = GPT(GPTConfig(**margs))
    sd = ckpt["model"]
    for k in list(sd):
        if k.startswith("_orig_mod."):
            sd[k[10:]] = sd.pop(k)
    model.load_state_dict(sd)
    model.to(device).eval()

    pairs = []
    with open(args.pairs, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(b"\t")
            if len(parts) != 3:
                continue
            label = int(parts[0]); A, B = parts[1], parts[2]
            pairs.append((label, A, B))
            if args.limit is not None and len(pairs) >= args.limit:
                break

    correct = 0
    for label, A, B in pairs:
        inpA, attA, typeA = encode_pair(A, B, cls_id, sep_id, pad_id, block_size)
        inpA, attA, typeA = inpA.to(device), attA.to(device), typeA.to(device)
        logits, _ = model(inpA, attention_mask=attA, token_type_ids=typeA, mode="cls")
        pred = 1 if logits.item() > 0 else 0  # 1 means A is original
        if pred == label:
            correct += 1

    total = len(pairs)
    acc = correct / total if total else 0.0
    print(f"pairs={total} correct={correct} accuracy={acc:.6f}")

if __name__ == "__main__":
    main()
