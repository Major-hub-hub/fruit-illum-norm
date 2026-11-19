"""Synthetic illumination stress transformations."""

import numpy as np


def dark(image: np.ndarray, factor: float = 0.45) -> np.ndarray:
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def bright(image: np.ndarray, factor: float = 1.5) -> np.ndarray:
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def colour_cast(image: np.ndarray, gains=(1.15, 1.0, 0.85)) -> np.ndarray:
    return np.clip(image.astype(np.float32) * np.asarray(gains), 0, 255).astype(np.uint8)

