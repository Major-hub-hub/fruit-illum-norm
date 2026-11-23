"""Sanity checks for all preprocessing pipelines."""

import numpy as np

from src.preprocess.pipeline import PIPELINES, apply_pipeline


def main():
    rng = np.random.default_rng(42)

    # Synthetic RGB image
    image = rng.integers(
        0, 256,
        size=(250, 250, 3),
        dtype=np.uint8
    )

    for condition in PIPELINES:
        result = apply_pipeline(image, condition)

        assert result.shape == image.shape
        assert result.dtype == np.uint8

        if condition == "B0":
            assert np.array_equal(result, image)

        changed = np.count_nonzero(image != result)

        print(
            f"{condition}: PASS | "
            f"Shape: {result.shape} | "
            f"Dtype: {result.dtype} | "
            f"Changed values: {changed}"
        )

    print("\nAll preprocessing checks passed!")


if __name__ == "__main__":
    main()