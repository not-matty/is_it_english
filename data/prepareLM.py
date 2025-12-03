import os
import random
import numpy as np
import pickle

random.seed(1337)

def main():
    current_directory = os.getcwd()
    in_path = os.path.join(current_directory, "data/train.txt")
    out_dir = os.path.join(current_directory, "data/train_lm")
    os.makedirs(out_dir, exist_ok=True)

    with open(in_path, "rb") as f:
        raw = f.read()

    pairs = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split(b"\t")
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))

    random.shuffle(pairs)
    n = len(pairs)
    split = int(n * 0.9)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    # token ids
    BOS, EOS, PAD = 256, 257, 258
    VOCAB_SIZE = 259

    def encode_bytes(b: bytes) -> np.ndarray:
        x = np.frombuffer(b, dtype=np.uint8)
        return np.concatenate(([BOS], x, [EOS])).astype(np.uint16)

    train_stream = []
    for o, _ in train_pairs:
        if len(o):
            train_stream.append(encode_bytes(o))
    val_stream = []
    for o, _ in val_pairs:
        if len(o):
            val_stream.append(encode_bytes(o))

    train_ids = np.concatenate(train_stream) if train_stream else np.array([], dtype=np.uint16)
    val_ids = np.concatenate(val_stream) if val_stream else np.array([], dtype=np.uint16)

    print("train tokens:", train_ids.shape[0], "val tokens:", val_ids.shape[0])

    train_ids.tofile(os.path.join(out_dir, "train.bin"))
    val_ids.tofile(os.path.join(out_dir, "val.bin"))

    meta = {
        "vocab_size": VOCAB_SIZE,
        "bos_id": BOS,
        "eos_id": EOS,
        "pad_id": PAD,
        "byte_encoding": "latin-1",
    }
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    with open(os.path.join(out_dir, "split_pairs.pkl"), "wb") as f:
        pickle.dump({"train": train_pairs, "val": val_pairs}, f)

if __name__ == "__main__":
    main()
