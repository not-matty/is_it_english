"""
Evaluation script for pair classification using an LM NLL comparison. Naive implementation (just compares NLL).
Usage example:
$ python classify.py out_lm/ckpt.pt [limit]
"""

import os, sys, pickle, torch, numpy as np
from LM.model import GPT, GPTConfig

# CLI args: checkpoint path (required) and optional limit on number of pairs
ckpt_path = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# standard weight-tying checkpoints sometimes carry an _orig_mod prefix; strip it like in train.py
ckpt = torch.load(ckpt_path, map_location=device)
margs = ckpt['model_args']; gpt = GPT(GPTConfig(**margs)); sd = ckpt['model']
for k in list(sd):
    if k.startswith('_orig_mod.'): sd[k[10:]] = sd.pop(k)
gpt.load_state_dict(sd); gpt.to(device).eval()

# expect meta with BOS/EOS from language-model data prep
meta = pickle.load(open(os.path.join('data','train_lm','meta.pkl'),'rb'))
BOS, EOS = meta['bos_id'], meta['eos_id']
# default eval pairs from train/val; optionally cut down via limit
eval_path = os.path.join('data','train','val_pairs.tsv')
pairs = []
with open(eval_path, 'rb') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(b'\t')
        if len(parts) != 3:
            continue
        label = int(parts[0])
        A, B = parts[1], parts[2]
        pairs.append((label, A, B))
if limit is not None:
    pairs = pairs[:limit]

bsz = getattr(gpt.config, 'block_size', 128)

@torch.no_grad()
def seq_nll(b: bytes) -> float:
    x = np.concatenate(([BOS], np.frombuffer(b, dtype=np.uint8), [EOS])).astype(np.int64)
    nll = 0.0
    for i in range(0, len(x), bsz):
        chunk = torch.from_numpy(x[i:i+bsz])
        if len(chunk) < 2: break
        X, Y = chunk[:-1].unsqueeze(0).to(device), chunk[1:].unsqueeze(0).to(device)
        _, loss = gpt(X, Y)
        nll += loss.item() * Y.numel()
    return nll

correct = 0
total = 0

# loop over pairs, lower NLL wins (predicts original)
for idx, (label, A, B) in enumerate(pairs):
    nA, nB = seq_nll(A), seq_nll(B)
    pred = 1 if nA < nB else 0  # 1 means A predicted original
    if pred == label:
        correct += 1
    total += 1
    # print(f"{idx}\t{nA:.6f}\t{nB:.6f}\t{pred}")

if total > 0:
    acc = correct / total
    print(f"# correct={correct}\ttotal={total}\taccuracy={acc:.6f}")
else:
    print("# No pairs evaluated.")
