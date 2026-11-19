"""Gray World colour constancy."""

import cv2
import numpy as np


def gray_world(image: np.ndarray) -> np.ndarray:
    """Scale RGB channels so their means become equal."""
    image = image.astype(np.float32)
    means = image.mean(axis=(0, 1), keepdims=True)
    target = means.mean()
    corrected = image * (target / (means + 1e-6))
    return np.clip(corrected, 0, 255).astype(np.uint8)

