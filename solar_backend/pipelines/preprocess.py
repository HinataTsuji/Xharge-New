"""Image preprocessing stages used before model inference."""
from __future__ import annotations

import cv2
import numpy as np


def preprocess_for_segmentation(image_bgr: np.ndarray) -> np.ndarray:
    """Prepare an image for roof segmentation.

    Uses contrast enhancement and denoising to make roof edges clearer.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(enhanced, d=7, sigmaColor=35, sigmaSpace=35)
