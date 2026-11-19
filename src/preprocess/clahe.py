"""CLAHE applied to the luminance channel in Lab colour space."""

import cv2
import numpy as np


def clahe_rgb(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    enhancer = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    result = cv2.merge((enhancer.apply(l_channel), a_channel, b_channel))
    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

