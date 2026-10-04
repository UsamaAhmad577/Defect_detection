# src/prepare_data.py
import argparse
from pathlib import Path
import cv2, numpy as np, pandas as pd, imagehash
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

LABELS = {"ok_front": 0, "def_front": 1}
EXTS = {".jpg", ".jpeg", ".png"}

def collect(raw):
    rows = []
    for p in Path(raw).rglob("*"):
        if p.suffix.lower() in EXTS and p.parent.name in LABELS:
            rows.append({"path": str(p), "label": LABELS[p.parent.name],
                         "orig_split": p.parent.parent.name})
    return pd.DataFrame(rows)

def group_ids(paths, thr):
    """Cluster images whose perceptual hashes are within `thr` bits -> same group."""
    hashes = np.array([int(str(imagehash.phash(Image.open(p))), 16) for p in paths],
                      dtype=np.uint64)
    n = len(hashes)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n - 1):
        x = hashes[i] ^ hashes[i + 1:]
        dist = np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(1)
        for j in np.where(dist <= thr)[0]:
            parent[find(i)] = find(i + 1 + j)
    return np.array([find(i) for i in range(n)])

def main(a):
    df = collect(a.raw)
    print("Original counts:\n", df.groupby(["orig_split", "label"]).size())

    # 1) Simulate realistic imbalance (keep all normal, subsample defective)
    if a.defect_ratio > 0:
        normal = df[df.label == 0]
        n_def = int(len(normal) * a.defect_ratio / (1 - a.defect_ratio))
        defect = df[df.label == 1].sample(n=min(n_def, (df.label == 1).sum()), random_state=42)
        df = pd.concat([normal, defect]).reset_index(drop=True)

    # 2) Leakage control: group near-duplicates, split by group
    df["group"] = group_ids(df.path, a.hash_thr)
    sizes = df.group.value_counts()
    print(f"Groups: {sizes.size} | largest group: {sizes.iloc[0]} images")
    # If the largest group is huge, hash_thr is too loose: lower it.

    # 3) ~72/14/14 stratified + grouped split (test first, then val)
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
    rest_idx, test_idx = next(sgkf.split(df, df.label, df.group))
    rest = df.iloc[rest_idx]
    sgkf2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)
    tr_idx, va_idx = next(sgkf2.split(rest, rest.label, rest.group))
    df["split"] = ""
    df.loc[df.index[test_idx], "split"] = "test"
    df.loc[rest.index[va_idx], "split"] = "val"
    df.loc[rest.index[tr_idx], "split"] = "train"

    print(df.groupby(["split", "label"]).size().unstack())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/splits.csv")
    ap.add_argument("--defect_ratio", type=float, default=0.12)  # 0 = keep original
    ap.add_argument("--hash_thr", type=int, default=2)
    main(ap.parse_args())