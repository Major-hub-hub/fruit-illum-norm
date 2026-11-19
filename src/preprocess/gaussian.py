"""Gaussian blur preprocessing."""

import cv2
import numpy as np


def gaussian_blur(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (5, 5), 0)

