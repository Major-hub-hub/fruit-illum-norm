"""Build the ONE fixed train/val/test split (80/10/10, stratified per class).

Usage (from repo root):
    python src/data/make_split.py --root data/raw --list     # inspect folders first
    python src/data/make_split.py --root data/raw            # write data/splits/split_seed42.csv

Assumptions (verify with --list!): the class is the image's parent folder name,
and that name contains 'fresh' or 'rotten' plus the fruit (e.g. 'freshapples').
Any path containing 'augment' is skipped: only ORIGINAL images are used,
otherwise augmented copies of one photo could land in train AND test (leakage).
"""
import argparse, re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

EXT = {".jpg", ".jpeg", ".png", ".bmp"}
FRUITS = ["apple", "banana", "grape", "guava", "jujube", "orange", "pomegranate", "strawberry"]


def collect(root: Path):
    rows = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in EXT:
            continue
        low = str(p.relative_to(root)).lower()
        if "augment" in low:
            continue
        cls = p.parent.name.lower()
        state = "rotten" if "rotten" in cls else "fresh" if "fresh" in cls else None
        fruit = next((f for f in FRUITS if f in cls), None)
        if state is None or fruit is None:
            continue
        rows.append(dict(path=p.as_posix(), label=int(state == "rotten"),  # 1 = rotten
                         fruit=fruit, cls=f"{state}_{fruit}"))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--out", default="data/splits/split_seed42.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--list", action="store_true", help="only print folder/image counts")
    a = ap.parse_args()
    root = Path(a.root)

    if a.list:
        counts = {}
        for p in root.rglob("*"):
            if p.suffix.lower() in EXT:
                counts[p.parent.relative_to(root).as_posix()] = counts.get(p.parent.relative_to(root).as_posix(), 0) + 1
        for k, v in sorted(counts.items()):
            print(f"{v:5d}  {k}")
        return

    df = collect(root)
    print("images found:", len(df), "| expected 3200")
    print(df.cls.value_counts().sort_index())
    assert df.cls.nunique() == 16, "expected 16 classes - run with --list and adjust the parsing"

    # stratified 80 / 10 / 10 per class
    train, rest = train_test_split(df, test_size=0.2, stratify=df.cls, random_state=a.seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest.cls, random_state=a.seed)
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.split.value_counts())  # expect 2560 / 320 / 320
    print("wrote", a.out)


if __name__ == "__main__":
    main()