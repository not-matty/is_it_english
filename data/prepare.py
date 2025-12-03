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


CLS, SEP, PAD, MASK = 256, 257, 258, 259
VOCAB_SIZE = 260
BLOCK_SIZE = 1024


meta = {
    "vocab_size": VOCAB_SIZE,
    "cls_id": CLS,
    "sep_id": SEP,
    "pad_id": PAD,
    "mask_id": MASK,
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
    max_len = BLOCK_SIZE - 3  # reserve CLS and two SEP

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
        trimmed_a = A[:max_len]
        trimmed_b = B[:max_len]
    else:
        diff_len = de - ds + 1
        budget = max_len - diff_len
        left_budget = budget // 2
        right_budget = budget - left_budget
        start_a = max(0, ds - left_budget)
        end_a = min(len(A), de + 1 + right_budget)
        trimmed_a = A[start_a:end_a]

        # align B span
        start_b = start_a
        end_b = end_a
        trimmed_b = B[start_b:end_b]

        # if still too long, trim equally
        if len(trimmed_a) > max_len:
            trimmed_a = trimmed_a[:max_len]
        if len(trimmed_b) > max_len:
            trimmed_b = trimmed_b[:max_len]

    ids = [CLS] + list(trimmed_a) + [SEP] + list(trimmed_b) + [SEP]
    ids = ids[:BLOCK_SIZE]
    types = [0] * (1 + len(trimmed_a) + 1) + [1] * (min(len(trimmed_b), BLOCK_SIZE - len(ids) + len(trimmed_b)) + 1)
    types = types[:len(ids)]
    attn = [1] * len(ids)
    if len(ids) < BLOCK_SIZE:
        pad_len = BLOCK_SIZE - len(ids)
        ids += [PAD] * pad_len
        types += [0] * pad_len
        attn += [0] * pad_len
    return (
        np.array(ids, np.uint16),
        np.array(attn, np.uint8),
        np.array(types, np.uint8),
    )

def build_npz(examples):
    inp, att, lab, types = [], [], [], []
    for label, A, B in examples:
        ids, mask, ttypes = encode_pair(A, B)
        inp.append(ids)
        att.append(mask)
        lab.append(label)
        types.append(ttypes)
    return dict(
        input_ids=np.stack(inp),
        attention_mask=np.stack(att),
        labels=np.array(lab, np.uint8),
        token_type_ids=np.stack(types),
    )

train_labeled = make_labeled_examples(train_pairs)
val_labeled   = make_labeled_examples(val_pairs)
np.savez_compressed(os.path.join(out_dir, "train_ce.npz"), **build_npz(train_labeled))
np.savez_compressed(os.path.join(out_dir, "val_ce.npz"), **build_npz(val_labeled))
write_pairs(os.path.join(out_dir, 'train_pairs.tsv'), train_labeled)
write_pairs(os.path.join(out_dir, 'val_pairs.tsv'), val_labeled)


def encode_original(orig: bytes):
    ids = [CLS, *orig, SEP][:BLOCK_SIZE]
    attn = [1] * len(ids)
    if len(ids) < BLOCK_SIZE:
        pad_len = BLOCK_SIZE - len(ids)
        ids += [PAD] * pad_len
        attn += [0] * pad_len
    return np.array(ids, np.uint16), np.array(attn, np.uint8)


def apply_mask(ids: np.ndarray):
    labels = np.full(ids.shape, fill_value=-100, dtype=np.int32)
    # candidate positions: non-special, non-pad
    candidates = np.where(
        (ids != CLS) & (ids != SEP) & (ids != PAD) & (ids != MASK)
    )[0]
    if candidates.size == 0:
        return ids, labels
    num_mask = max(1, int(0.15 * candidates.size))
    chosen = np.random.choice(candidates, size=num_mask, replace=False)
    for idx in chosen:
        labels[idx] = ids[idx]
        rnd = random.random()
        if rnd < 0.8:
            ids[idx] = MASK
        elif rnd < 0.9:
            ids[idx] = random.randint(0, 255)
        else:
            pass  # keep original token
    return ids, labels


def build_mlm_npz(pairs_list):
    inputs, attns, labels = [], [], []
    for orig, _ in pairs_list:
        ids, attn = encode_original(orig)
        ids, lbl = apply_mask(ids)
        inputs.append(ids)
        attns.append(attn)
        labels.append(lbl)
    return dict(
        input_ids=np.stack(inputs),
        attention_mask=np.stack(attns),
        labels=np.stack(labels),
    )


np.savez_compressed(os.path.join(out_dir, "train_mlm.npz"), **build_mlm_npz(train_pairs))
np.savez_compressed(os.path.join(out_dir, "val_mlm.npz"), **build_mlm_npz(val_pairs))
