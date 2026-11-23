"""Gray World white balancing for fruit image preprocessing.

Experiment E1: Gray World
Experiment E3: Gray World + CLAHE

Corrects global colour imbalance using the Gray World
assumption that the average scene colour should be neutral.
"""

import numpy as np


def gray_world(image: np.ndarray) -> np.ndarray:
    """Apply Gray World white balancing to an RGB image.

    Args:
        image: RGB uint8 image with shape (H, W, 3).

    Returns:
        White-balanced RGB uint8 image with the same shape.
    """

    # 1. Validate input
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected an RGB image with shape (H, W, 3)."
        )

    if image.dtype != np.uint8:
        raise TypeError("Image must have dtype uint8.")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Image dimensions must be positive.")

    # 2. Convert to floating point for calculations
    img_float = image.astype(np.float32)

    # 3. Calculate average intensity of each RGB channel
    channel_means = img_float.mean(axis=(0, 1))

    # 4. Calculate target gray intensity
    target_mean = channel_means.mean()

    # 5. Calculate correction factors safely
    gains = np.ones(3, dtype=np.float32)

    valid_channels = channel_means > 1e-6

    gains[valid_channels] = (
        target_mean / channel_means[valid_channels]
    )

    # 6. Apply correction to each RGB channel
    corrected = img_float * gains.reshape(1, 1, 3)

    # 7. Clip pixel values and restore uint8 format
    corrected = np.clip(corrected, 0, 255)

    return corrected.astype(np.uint8)