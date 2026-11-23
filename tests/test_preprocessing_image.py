"""Unit-check preprocessing functions on one real fruit image.

Run from the repository root:
    python -m tests.test_preprocessing_image

If tests/ is not a Python package, use:
    python tests/test_preprocessing_image.py
"""

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess.gaussian import gaussian_blur
from src.preprocess.gray_world import gray_world
from src.preprocess.clahe import clahe_rgb
from src.preprocess.pipeline import PIPELINES, apply_pipeline


def check_image(name, original, result):
    """Check basic image properties and input preservation."""

    assert isinstance(result, np.ndarray), name
    assert result.shape == (250, 250, 3), name
    assert result.dtype == np.uint8, name
    assert np.array_equal(original, original_backup), (
        f"{name} modified the input image!"
    )

    changed = np.count_nonzero(original != result)
    difference = np.mean(
        np.abs(
            original.astype(np.float32)
            - result.astype(np.float32)
        )
    )

    print(
        f"{name:20s} PASS | "
        f"Changed: {changed:6d} | "
        f"Mean difference: {difference:.3f}"
    )


# STEP 1: Select one image from the TRAIN split
manifest = pd.read_csv(
    ROOT / "data/splits/split_seed42.csv"
)
row = manifest.loc[manifest["split"] == "train"].iloc[0]

image_path = ROOT / str(row["path"])

# Resize using the same dimensions and interpolation
# as FruitDataset's PIL Resize
with Image.open(image_path) as img:
    original = np.asarray(
        img.convert("RGB").resize(
            (250, 250),
            resample=Image.Resampling.BILINEAR
        )
    )

original_backup = original.copy()

print("Testing image:", image_path)
print("Shape:", original.shape)
print("Dtype:", original.dtype)
print()

# STEP 2: Test Gaussian
gaussian = gaussian_blur(original)
check_image("Gaussian", original, gaussian)

# Verify against the OpenCV reference operation
assert np.array_equal(
    gaussian,
    cv2.GaussianBlur(original, (5, 5), 0)
)

# STEP 3: Test Gray World
gray = gray_world(original)
check_image("Gray World", original, gray)

print("Original RGB means:",
      original.mean(axis=(0, 1)).round(2))
print("Gray World RGB means:",
      gray.mean(axis=(0, 1)).round(2))

# STEP 4: Test CLAHE
clahe = clahe_rgb(original)
check_image("CLAHE", original, clahe)

# Verify the CLAHE implementation independently
lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)
reference_clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)
expected_lab = cv2.merge(
    (reference_clahe.apply(l), a, b)
)
expected_rgb = cv2.cvtColor(
    expected_lab, cv2.COLOR_LAB2RGB
)
assert np.array_equal(clahe, expected_rgb)

# STEP 5: Test all registered pipelines
results = {}

for condition in PIPELINES:
    result = apply_pipeline(original, condition)
    check_image(condition, original, result)
    results[condition] = result

# Verify B0 leaves pixel values unchanged
assert np.array_equal(results["B0"], original)

# Verify each pipeline's intended order
assert np.array_equal(results["B1"], gaussian)
assert np.array_equal(results["E1"], gray)
assert np.array_equal(results["E2"], clahe)

assert np.array_equal(
    results["E3"],
    clahe_rgb(gray_world(original))
)

assert np.array_equal(
    results["E4"],
    gaussian_blur(clahe_rgb(gray_world(original)))
)

# STEP 6: Save visual comparison
out_dir = ROOT / "outputs/figures"
out_dir.mkdir(parents=True, exist_ok=True)

names = {
    "B0": "Original",
    "B1": "Gaussian 5x5",
    "E1": "Gray World",
    "E2": "CLAHE",
    "E3": "Gray World + CLAHE",
    "E4": "Gray World + CLAHE + Gaussian"
}

fig, axes = plt.subplots(2, 3, figsize=(13, 9))

for ax, (condition, image) in zip(
    axes.flat, results.items()
):
    ax.imshow(image)
    ax.set_title(
        f"{condition}: {names[condition]}"
    )
    ax.axis("off")

fig.suptitle(
    "Fruit Preprocessing Comparison",
    fontsize=16
)
fig.tight_layout()

output_path = out_dir / "preprocessing_comparison.png"
fig.savefig(output_path, dpi=180)
plt.close(fig)

print()
print("All preprocessing checks PASSED!")
print("Saved comparison:", output_path)