"""Deterministic synthetic illumination perturbations for held-out RGB images.

Apply after resizing and before model-specific preprocessing and tensor
normalisation. Inputs and outputs are H x W x 3 RGB uint8 arrays.
"""
from collections.abc import Callable
import numpy as np

# Match the existing evaluate.py settings.
DARK_FACTOR = 0.50
BRIGHT_FACTOR = 1.40
WARM_GAINS = (1.20, 1.00, 0.75)
STRESSES = ("normal", "dark", "bright", "warm_cast")


def _validate(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("Expected a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError("Expected a non-empty H x W x 3 RGB image")
    if image.dtype != np.uint8:
        raise TypeError("Expected uint8 RGB image")


def _scale(image: np.ndarray, gains) -> np.ndarray:
    _validate(image)
    gains = np.asarray(gains, dtype=np.float32)
    if not np.isfinite(gains).all() or np.any(gains <= 0):
        raise ValueError("Gains must be finite and positive")
    return np.clip(image.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def normal(image: np.ndarray) -> np.ndarray:
    _validate(image)
    return image.copy()


def dark(image: np.ndarray, factor: float = DARK_FACTOR) -> np.ndarray:
    if not 0 < factor < 1:
        raise ValueError("Dark factor must be between 0 and 1")
    return _scale(image, factor)


def bright(image: np.ndarray, factor: float = BRIGHT_FACTOR) -> np.ndarray:
    if not factor > 1:
        raise ValueError("Bright factor must exceed 1")
    return _scale(image, factor)


def colour_cast(image: np.ndarray, gains=WARM_GAINS) -> np.ndarray:
    if np.asarray(gains).shape != (3,):
        raise ValueError("Expected three gains in RGB order")
    return _scale(image, gains)


STRESS_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "normal": normal,
    "dark": dark,
    "bright": bright,
    "warm_cast": colour_cast,
}


def apply_stress(image: np.ndarray, stress: str = "normal") -> np.ndarray:
    """Apply a named lighting condition without modifying the input."""
    if stress not in STRESS_TRANSFORMS:
        raise ValueError(f"Unknown stress {stress!r}; choose from {STRESSES}")
    return STRESS_TRANSFORMS[stress](image)


if __name__ == "__main__":
    image = np.array([[[100, 100, 100], [250, 200, 10]]], dtype=np.uint8)
    original = image.copy()
    expected = {
        "normal": [100, 100, 100],
        "dark": [50, 50, 50],
        "bright": [140, 140, 140],
        "warm_cast": [120, 100, 75],
    }
    for name in STRESSES:
        result = apply_stress(image, name)
        assert result.shape == image.shape and result.dtype == np.uint8
        assert result[0, 0].tolist() == expected[name]
        assert np.array_equal(result, apply_stress(image, name))
        assert np.array_equal(image, original)
        print(f"{name}: {result.tolist()}")
    assert apply_stress(image, "bright")[0, 1, 0] == 255
    print("All stress transformation checks passed.")
