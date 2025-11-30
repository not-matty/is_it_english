import os, random, pickle
import numpy as np
random.seed(1337)

current_directory = os.getcwd()
in_path  = os.path.join(current_directory, 'data/train.txt')
out_dir  = os.path.join(current_directory, 'data/train')
os.makedirs(out_dir, exist_ok=True)

with open(in_path, 'rb') as f:
    raw = f.read()


pairs = []
for line in raw.splitlines():
    if not line.strip(): 
        continue
    parts = line.split(b'\t')
    if len(parts) == 2:
        pairs.append((parts[0], parts[1]))

random.shuffle(pairs)
n = len(pairs)
split = int(n * 0.9)
train_pairs = pairs[:split]
val_pairs   = pairs[split:]


CLS, SEP, PAD = 256, 257, 258
VOCAB_SIZE = 259
BLOCK_SIZE = 1024


meta = {
    "vocab_size": VOCAB_SIZE,
    "cls_id": CLS,
    "sep_id": SEP,
    "pad_id": PAD,
    "block_size": BLOCK_SIZE,
    "byte_encoding": "latin-1",
}

with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

with open(os.path.join(out_dir, 'split_pairs.pkl'), 'wb') as f:
    pickle.dump({"train": train_pairs, "val": val_pairs}, f)


def make_labeled_examples(pairs):
    labeled = []
    for o, a in pairs:
        if random.random() < 0.5:
            labeled.append((1, o, a))  # A is original
        else:
            labeled.append((0, a, o))  # B is original
    return labeled


def write_pairs(path, examples):
    with open(path, 'wb') as f:
        for label, A, B in examples:
            f.write(str(label).encode('ascii') + b'\t' + A + b'\t' + B + b'\n')

def encode_pair(A: bytes, B: bytes):
    ids = [CLS, *A, SEP, *B, SEP][:BLOCK_SIZE]
    attn = [1] * len(ids)
    if len(ids) < BLOCK_SIZE:
        pad_len = BLOCK_SIZE - len(ids)
        ids += [PAD] * pad_len
        attn += [0] * pad_len
    return np.array(ids, np.uint16), np.array(attn, np.uint8)

def build_npz(examples):
    inp, att, lab = [], [], []
    for label, A, B in examples:
        ids, mask = encode_pair(A, B)
        inp.append(ids); att.append(mask); lab.append(label)
    return dict(input_ids=np.stack(inp), attention_mask=np.stack(att), labels=np.array(lab, np.uint8))

train_labeled = make_labeled_examples(train_pairs)
val_labeled   = make_labeled_examples(val_pairs)
np.savez_compressed(os.path.join(out_dir, "train_ce.npz"), **build_npz(train_labeled))
np.savez_compressed(os.path.join(out_dir, "val_ce.npz"), **build_npz(val_labeled))
write_pairs(os.path.join(out_dir, 'train_pairs.tsv'), train_labeled)
write_pairs(os.path.join(out_dir, 'val_pairs.tsv'), val_labeled)
