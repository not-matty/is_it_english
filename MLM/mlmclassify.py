import argparse
import os
import pickle
import torch
import numpy as np
import random

from model import GPT, GPTConfig

random.seed(1337)


def load_meta(meta_path):
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def encode_sequence(seq: bytes, cls_id: int, sep_id: int, pad_id: int, block_size: int):
    ids = [cls_id]
    for b in seq:
        if len(ids) >= block_size - 1:
            break
        ids.append(b)
    ids.append(sep_id)
    ids = ids[:block_size]
    attn = [1] * len(ids)
    pad_len = 0
    if len(ids) < block_size:
        pad_len = block_size - len(ids)
        ids += [pad_id] * pad_len
        attn += [0] * pad_len
    token_type_ids = [0] * len(ids)  # single segment
    return (
        torch.tensor([ids], dtype=torch.long),
        torch.tensor([attn], dtype=torch.long),
        torch.tensor([token_type_ids], dtype=torch.long),
        len(ids) - pad_len,
    )


def apply_mask(ids: torch.Tensor, cls_id: int, sep_id: int, pad_id: int, mask_id: int):
    # ids shape: (1, T)
    ids = ids.clone()
    labels = torch.full_like(ids, fill_value=-100)
    valid = (ids != cls_id) & (ids != sep_id) & (ids != pad_id) & (ids != mask_id)
    positions = valid.nonzero(as_tuple=False).reshape(-1)
    if positions.numel() == 0:
        return ids, labels, 0
    num_mask = max(1, int(0.15 * positions.numel()))
    chosen = positions[torch.randperm(len(positions))[:num_mask]]
    for idx in chosen:
        labels[0, idx] = ids[0, idx]
        rnd = random.random()
        if rnd < 0.8:
            ids[0, idx] = mask_id
        elif rnd < 0.9:
            ids[0, idx] = random.randint(0, 255)
        else:
            pass
    return ids, labels, chosen.numel()


@torch.no_grad()
def seq_total_loss(model, seq_bytes, ids):
    cls_id, sep_id, pad_id, mask_id, block_size = ids
    input_ids, attn_mask, type_ids, real_len = encode_sequence(seq_bytes, cls_id, sep_id, pad_id, block_size)
    input_ids, labels, masked_count = apply_mask(input_ids, cls_id, sep_id, pad_id, mask_id)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    type_ids = type_ids.to(device)
    labels = labels.to(device)
    logits, loss = model(input_ids, labels=labels, attention_mask=attn_mask, token_type_ids=type_ids, mode="mlm")
    # loss is mean over masked positions; scale by masked_count to compare totals
    if masked_count == 0:
        return float("inf")
    return loss.item() * masked_count


def main():
    parser = argparse.ArgumentParser(description="Quick MLM loss comparison on labeled pairs.")
    parser.add_argument("ckpt", help="Path to checkpoint (ckpt.pt)")
    parser.add_argument("--pairs", default=os.path.join("data", "train", "val_pairs.tsv"), help="TSV with label\\tA\\tB")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of pairs")
    parser.add_argument("--meta", default=os.path.join("data", "train", "meta.pkl"), help="Path to meta.pkl")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = load_meta(args.meta)
    cls_id = meta["cls_id"]
    sep_id = meta["sep_id"]
    pad_id = meta["pad_id"]
    mask_id = meta["mask_id"]
    block_size = meta["block_size"]

    ckpt = torch.load(args.ckpt, map_location=device)
    margs = ckpt["model_args"]
    gpt = GPT(GPTConfig(**margs))
    sd = ckpt["model"]
    for k in list(sd):
        if k.startswith("_orig_mod."):
            sd[k[10:]] = sd.pop(k)
    gpt.load_state_dict(sd)
    gpt.to(device).eval()

    pairs = []
    with open(args.pairs, "rb") as f:
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
            if args.limit is not None and len(pairs) >= args.limit:
                break

    correct = 0
    for label, A, B in pairs:
        loss_a = seq_total_loss(gpt, A, (cls_id, sep_id, pad_id, mask_id, block_size))
        loss_b = seq_total_loss(gpt, B, (cls_id, sep_id, pad_id, mask_id, block_size))
        pred = 1 if loss_a < loss_b else 0  # 1 means A is original
        if pred == label:
            correct += 1

    total = len(pairs)
    acc = correct / total if total else 0.0
    print(f"pairs={total} correct={correct} accuracy={acc:.6f}")


if __name__ == "__main__":
    main()
