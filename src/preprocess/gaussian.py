"""Gaussian filtering for fresh–rotten fruit image preprocessing.

B1 uses a 5×5 Gaussian kernel, following the simplified
reproduction of Fischer-Brandies et al. (2025).
"""

import cv2
import numpy as np


def gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0.0
) -> np.ndarray:
    """Apply Gaussian smoothing to an RGB image.

    Args:
        image: RGB image as a uint8 NumPy array (H, W, 3).
        kernel_size: Odd, positive Gaussian kernel size.
                     Default is 5 for experiment B1.
        sigma: Gaussian standard deviation.
               0 lets OpenCV calculate it automatically.

    Returns:
        Gaussian-blurred RGB image with unchanged shape and dtype.
    """

    # Validate image format
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image of shape (H, W, 3).")

    if image.dtype != np.uint8:
        raise TypeError("Image must have dtype uint8.")

    # Validate kernel size
    if (
        not isinstance(kernel_size, int)
        or isinstance(kernel_size, bool)
        or kernel_size <= 0
        or kernel_size % 2 == 0
    ):
        raise ValueError("Kernel size must be a positive odd integer.")

    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Sigma must be finite and non-negative.")

    # Apply Gaussian filter
    blurred = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=float(sigma),
        sigmaY=float(sigma)
    )

    return blurred