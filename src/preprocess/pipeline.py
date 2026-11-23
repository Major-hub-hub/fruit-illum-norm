"""Preprocessing pipeline registry for fruit classification.

Experimental conditions:
    B0: Original RGB (no preprocessing)
    B1: Gaussian blur (5x5)
    E1: Gray World white balancing
    E2: CLAHE
    E3: Gray World -> CLAHE
    E4: Gray World -> CLAHE -> Gaussian (5x5)

All pipelines accept and return RGB uint8 images.
"""

from collections.abc import Callable

import numpy as np

from .gaussian import gaussian_blur
from .gray_world import gray_world
from .clahe import clahe_rgb


def identity(image: np.ndarray) -> np.ndarray:
    """B0: Return an unchanged copy of the RGB image."""
    return image.copy()


def gray_world_clahe(image: np.ndarray) -> np.ndarray:
    """E3: Apply Gray World followed by CLAHE."""
    corrected = gray_world(image)
    enhanced = clahe_rgb(corrected)
    return enhanced


def gray_world_clahe_gaussian(
    image: np.ndarray
) -> np.ndarray:
    """E4: Apply Gray World, CLAHE, then Gaussian 5x5."""
    corrected = gray_world(image)
    enhanced = clahe_rgb(corrected)
    smoothed = gaussian_blur(enhanced, kernel_size=5)
    return smoothed


# Registry used by the dataset loader and training scripts.
PIPELINES: dict[
    str, Callable[[np.ndarray], np.ndarray]
] = {
    "B0": identity,
    "B1": gaussian_blur,
    "E1": gray_world,
    "E2": clahe_rgb,
    "E3": gray_world_clahe,
    "E4": gray_world_clahe_gaussian,
}


def apply_pipeline(
    image: np.ndarray,
    condition: str = "B0"
) -> np.ndarray:
    """Validate an image and apply a selected pipeline.

    Args:
        image: RGB uint8 image with shape (H, W, 3).
        condition: Experimental condition (B0 to E4).

    Returns:
        Preprocessed RGB uint8 image.
    """

    if condition not in PIPELINES:
        raise ValueError(
            f"Unknown condition: {condition}. "
            f"Available: {list(PIPELINES.keys())}"
        )

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected RGB image with shape (H, W, 3)."
        )

    if image.dtype != np.uint8:
        raise TypeError("Image must have dtype uint8.")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Image dimensions must be positive.")

    result = PIPELINES[condition](image)

    if result.shape != image.shape:
        raise RuntimeError(
            f"{condition}: Output image shape changed."
        )

    if result.dtype != np.uint8:
        raise RuntimeError(
            f"{condition}: Output dtype must be uint8."
        )

    return result