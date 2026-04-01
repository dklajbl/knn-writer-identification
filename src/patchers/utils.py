import cv2
import numpy as np


def is_empty_or_padded_patch(
    patch: np.ndarray,
    zero_threshold: float = 0.99,
    std_threshold: float = 2.0,
) -> bool:

    """
    Decide whether a patch is effectively empty / padded.

    A patch is rejected if:
        - it is almost entirely zeros
        - or it is nearly constant
    Because before augmentation, padded areas are usually exactly zero and after augmentation, padded areas may become nearly constant but not zero.

    Parameters:
        patch (np.ndarray): Patch of shape (H, W, C).
        zero_threshold (float): Fraction of zero pixels above which patch is rejected.
        std_threshold (float): Standard deviation below which patch is treated as almost constant.

    Returns:
        bool: True if the patch should be rejected, False otherwise.
    """

    if patch.size == 0:
        return True

    # exact / near-zero padding detection
    zero_fraction = np.mean(patch == 0)
    if zero_fraction >= zero_threshold:
        return True

    # almost constant region detection - useful when augmentation changes padding from pure zero to some flat value
    if np.std(patch.astype(np.float32)) < std_threshold:
        return True

    return False


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

    return patch
