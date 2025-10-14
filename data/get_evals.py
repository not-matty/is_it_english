import pickle, os

pkl_path = os.path.join("train", "split_pairs.pkl")
with open(pkl_path, "rb") as f:
    splits = pickle.load(f)

train_pairs = splits["train"]
eval_pairs  = splits["val"]

print(f"train pairs: {len(train_pairs):,}, eval pairs: {len(eval_pairs):,}")

with open("eval_pairs.tsv", "wb") as f:
    for A, B in eval_pairs:
        f.write(A); f.write(b"\t"); f.write(B); f.write(b"\n")
