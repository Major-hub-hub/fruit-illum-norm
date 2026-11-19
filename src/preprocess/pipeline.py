"""Registry of preprocessing conditions."""

from collections.abc import Callable
import numpy as np

from .clahe import clahe_rgb
from .gaussian import gaussian_blur
from .gray_world import gray_world


def identity(image: np.ndarray) -> np.ndarray:
    return image


PIPELINES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "B0": identity,
    "B1": gaussian_blur,
    "E1": gray_world,
    "E2": clahe_rgb,
    "E3": lambda image: clahe_rgb(gray_world(image)),
}

