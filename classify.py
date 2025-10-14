# eval_pairs.py — minimal byte-level pair scorer with B/total summary and optional limit
import os, sys, pickle, torch, numpy as np
from model import GPT, GPTConfig

# --- args ---
if len(sys.argv) < 2:
    sys.exit("usage: python eval_pairs.py <ckpt.pt> [limit]")
ckpt_path = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- load model ---
ckpt = torch.load(ckpt_path, map_location=device)
margs = ckpt['model_args']; gpt = GPT(GPTConfig(**margs)); sd = ckpt['model']
for k in list(sd):
    if k.startswith('_orig_mod.'): sd[k[10:]] = sd.pop(k)
gpt.load_state_dict(sd); gpt.to(device).eval()

# --- meta & pairs ---
meta = pickle.load(open(os.path.join('data','train','meta.pkl'),'rb'))
BOS, EOS = meta['bos_id'], meta['eos_id']
eval_path = os.path.join('data','eval_pairs.tsv')
pairs = []
with open(eval_path, 'rb') as f:
    for line in f:
        line = line.strip()
        if line:
            pairs.append(line)
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

count_b = 0
total = 0

for idx, line in enumerate(pairs):
    A, B = line.split(b'\t', 1)
    nA, nB = seq_nll(A), seq_nll(B)
    pred = 'A' if nA < nB else 'B'
    if pred == 'A': count_b += 1
    total += 1
    #print(f"{idx}\t{nA:.6f}\t{nB:.6f}\t{pred}")

# --- summary ---
if total > 0:
    acc = count_b / total
    print(f"# A={count_b}\ttotal={total}\taccuracy(A/total)={acc:.6f}")
else:
    print("# No pairs evaluated.")
