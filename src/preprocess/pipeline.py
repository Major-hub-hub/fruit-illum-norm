"""Registry of preprocessing conditions."""

from collections.abc import Callable
import numpy as np

from .clahe import clahe_rgb
from .gaussian import gaussian_blur
from .gray_world import gray_world


def identity(image: np.ndarray) -> np.ndarray:
    return image


def gray_world_clahe(image: np.ndarray) -> np.ndarray:
    """Apply Gray World followed by CLAHE; usable by spawned loader workers."""
    return clahe_rgb(gray_world(image))


PIPELINES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "B0": identity,
    "B1": gaussian_blur,
    "E1": gray_world,
    "E2": clahe_rgb,
    "E3": gray_world_clahe,
}

