import cv2
import numpy as np


def normalize_patch_size(patch: np.ndarray, target_height: int, target_width: int, interpolation: int) -> np.ndarray:

    """
    Resize patch to the output size if needed.

    Parameters:
        patch (np.ndarray): Input patch
        target_height (int): Height of the output patch
        target_width (int): Width of the output patch
        interpolation (int): Interpolation mode

    Returns:
        np.ndarray: Patch with shape (patch_height, patch_width, C).
    """

    if patch.shape[:2] != (target_height, target_width):
        patch = cv2.resize(
            patch,
            (target_width, target_height),
            interpolation=interpolation
        )

    # cv2.resize drops the channel dim when C=1; restore it
    if patch.ndim == 2:
        patch = np.expand_dims(patch, axis=-1)

    return patch
