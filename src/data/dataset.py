"""Fixed-manifest PyTorch dataset and Friday data-pipeline smoke check.

Run from the repository root: python -m src.data.dataset
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.preprocess.pipeline import PIPELINES

ROOT = Path(__file__).resolve().parents[2]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def validate_manifest(csv_path, root=ROOT, check_files=True):
    """Validate the entire manifest before selecting any individual split."""
    root = Path(root)
    frame = pd.read_csv(csv_path)
    required = {"path", "label", "fruit", "cls", "split", "sha256"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("Empty manifest or missing required columns")
    if frame[list(required)].isna().any().any():
        raise ValueError("Missing manifest values")
    if set(frame.split) != {"train", "val", "test"} or not frame.label.isin([0, 1]).all():
        raise ValueError("Invalid split or label")
    if not frame.sha256.str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Invalid content hash")
    expected = frame.label.map({0: "fresh", 1: "rotten"}) + "_" + frame.fruit
    if not frame.cls.eq(expected).all():
        raise ValueError("Conflicting class and label")
    paths = [str((root / str(p)).resolve()) for p in frame.path]
    if len(set(paths)) != len(paths):
        raise ValueError("Repeated file path in manifest")
    if frame.groupby("sha256").split.nunique().gt(1).any():
        raise ValueError("Exact-content leakage across splits")
    if frame.groupby("sha256").cls.nunique().gt(1).any():
        raise ValueError("Conflicting labels for identical content")
    if check_files:
        missing = [p for p in paths if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} missing images; first: {missing[0]}")
    frame = frame.copy()
    frame["resolved_path"] = paths
    return frame


class FruitDataset(Dataset):
    """Return normalized RGB tensors and integer binary labels."""
    def __init__(self, csv_path=ROOT / "data/splits/split_seed42.csv",
                 split="train", condition="B0", img_size=250, root=ROOT,
                 augment=None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        if condition not in PIPELINES:
            raise ValueError(f"Unknown condition: {condition}")
        if augment and split != "train":
            raise ValueError("Augmentation is only allowed for training")
        frame = validate_manifest(csv_path, root)
        self.rows = frame.loc[frame.split == split].reset_index(drop=True)
        self.preprocess = PIPELINES[condition]
        self.resize = transforms.Resize((img_size, img_size),
                                        interpolation=transforms.InterpolationMode.BILINEAR)
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
        ]) if split == "train" and augment is not False else None
        self.tensor = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(row.resolved_path) as image:
            rgb = np.array(self.resize(image.convert("RGB")))
        image = Image.fromarray(self.preprocess(rgb))
        if self.augmentation is not None:
            image = self.augmentation(image)
        return self.tensor(image), int(row.label)


def main():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(ROOT / "data/splits/split_seed42.csv"))
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--condition", choices=PIPELINES, default="B0")
    parser.add_argument("--out", default=str(ROOT / "outputs/figures/dataset_sanity_grid.png"))
    args = parser.parse_args()
    torch.manual_seed(42)
    frame = validate_manifest(args.csv, args.root)
    counts = pd.crosstab(frame.cls, frame.split)
    if len(frame) != 3200 or len(counts) != 16 or not (
        counts["train"].eq(160) & counts["val"].eq(20) & counts["test"].eq(20)
    ).all():
        raise ValueError("Unexpected official dataset quotas")
    print("Manifest: no repeated paths or cross-split SHA-256 groups")
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    for r, split in enumerate(("train", "val", "test")):
        dataset = FruitDataset(args.csv, split, args.condition, root=args.root)
        loader = DataLoader(dataset, batch_size=32, shuffle=True,
                            generator=torch.Generator().manual_seed(42), num_workers=0)
        images, labels = next(iter(loader))
        if images.shape != (32, 3, 250, 250) or labels.shape != (32,):
            raise RuntimeError(f"Unexpected batch shape for {split}")
        if not torch.isfinite(images).all() or labels.dtype != torch.long:
            raise RuntimeError("Invalid tensor values or label dtype")
        print(f"{split}: {len(dataset)} files | {tuple(images.shape)} | {tuple(labels.shape)}")
        for c, ax in enumerate(axes[r]):
            image = images[c] * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]
            ax.imshow(image.permute(1, 2, 0).clamp(0, 1).numpy())
            ax.set_title(f"{split}: {'Rotten' if labels[c] else 'Fresh'}", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
