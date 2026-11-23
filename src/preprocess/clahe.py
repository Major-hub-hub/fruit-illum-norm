"""CLAHE preprocessing for fresh–rotten fruit classification.

Experiment E2: CLAHE
Experiment E3: Gray World + CLAHE

CLAHE (Contrast Limited Adaptive Histogram Equalization)
enhances local contrast while limiting noise amplification.

The algorithm is applied only to the L (lightness) channel
in CIELAB colour space to better preserve colour information.
"""

import cv2
import numpy as np


def clahe_rgb(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8
) -> np.ndarray:
    """Apply CLAHE to the lightness channel of an RGB image.

    Args:
        image: RGB uint8 image of shape (H, W, 3).
        clip_limit: Contrast clipping threshold.
                    Default is 2.0.
        grid_size: Number of tiles along each image dimension.
                   Default is 8.

    Returns:
        Contrast-enhanced RGB uint8 image with unchanged shape.
    """

    # 1. Validate input image
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

    # 2. Validate parameters
    if not np.isfinite(clip_limit) or clip_limit <= 0:
        raise ValueError("clip_limit must be finite and positive.")

    if (
        not isinstance(grid_size, int)
        or isinstance(grid_size, bool)
        or grid_size <= 0
    ):
        raise ValueError("grid_size must be a positive integer.")

    # 3. Convert RGB image to CIELAB colour space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # 4. Separate lightness and colour channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 5. Create CLAHE processor
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(grid_size, grid_size)
    )

    # 6. Apply CLAHE only to lightness
    enhanced_l = clahe.apply(l_channel)

    # 7. Recombine the Lab channels
    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    # 8. Convert enhanced Lab image back to RGB
    enhanced_rgb = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2RGB
    )

    return enhanced_rgb