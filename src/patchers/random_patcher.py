import numpy as np
from src.patchers.base_patcher import BasePatcher
from src.patchers.patcher_config import PatcherConfig
from src.patchers.utils import normalize_patch_size


class RandomPatcher(BasePatcher):

    """
    Random patcher for text and document images.

    This patcher extracts exactly "patch_count" random patches of fixed size (patch_height, patch_width) from the input image.

    Behavior:
        1. Generates all coordinates at once (vectorized RNG).
        2. Extracts all patches in bulk via numpy advanced indexing.

    Main properties:
        - returns exactly `patch_count` patches
        - uses explicit patch size from config
        - samples random valid crop positions
        - all output patches have the same shape
    """

    def __init__(self, config: PatcherConfig):

        """
        Initialize the random patcher.

        Parameters:
            config (PatcherConfig): Patcher configuration.

        Raises:
            ValueError: If the input patch size is invalid.
        """

        super().__init__(config)

        self.patch_height = config.patch_height
        self.patch_width = config.patch_width
        self.interpolation = config.interpolation

        if self.patch_height <= 0 or self.patch_width <= 0:
            raise ValueError("patch_height and patch_width must be > 0")

    def extract_patches(self, image: np.ndarray, key: str | None = None) -> np.ndarray:

        """
        Extract random patches from the input image.

        Parameters:
            image (np.ndarray): Input image of shape (H, W, C).
            key (str | None): ignored. Present to satisfy the BasePatcher interface.

        Returns:
            np.ndarray: Array of shape (patch_count, patch_height, patch_width, C).

        Raises:
            ValueError: If the input image is invalid.
        """

        if image is None:
            raise ValueError("image is None")

        if image.ndim != 3:
            raise ValueError(f"Expected image shape (H, W, C), got {image.shape}")

        # SPECIAL CASE: patch_count == 1 -> return full image (with expanded shape "patch_count"=1)
        if self.patch_count == 1:
            return np.expand_dims(image, axis=0)

        img_height, img_width, channels = image.shape

        # if the requested patch is larger than the image - crop what is available and resize to the target size
        crop_height, crop_width = min(self.patch_height, img_height), min(self.patch_width, img_width)

        max_top, max_left = max(0, img_height - crop_height), max(0, img_width - crop_width)
        needs_resize = (crop_height != self.patch_height or crop_width != self.patch_width)

        # generate all coordinates at once (vectorized RNG instead of per-patch calls)
        patch_tops = self.rng.integers(0, max_top + 1, size=self.patch_count) if max_top > 0 else np.zeros(self.patch_count, dtype=np.int64)
        patch_lefts = self.rng.integers(0, max_left + 1, size=self.patch_count) if max_left > 0 else np.zeros(self.patch_count, dtype=np.int64)

        # extract all patches via numpy indexing
        row_offsets, col_offsets = np.arange(crop_height), np.arange(crop_width)
        row_indices = patch_tops[:, None] + row_offsets[None, :]   # (patch_count, crop_height)
        col_indices = patch_lefts[:, None] + col_offsets[None, :]  # (patch_count, crop_width)
        patches = image[row_indices[:, :, None], col_indices[:, None, :], :]  # (patch_count, crop_height, crop_width, channels)

        # resize only when image is smaller than requested patch size (rare edge case)
        if needs_resize:
            resized_patches = np.empty((self.patch_count, self.patch_height, self.patch_width, channels), dtype=patches.dtype)
            for i in range(self.patch_count):
                resized_patches[i] = normalize_patch_size(patches[i], self.patch_height, self.patch_width, self.interpolation)
            patches = resized_patches

        return patches
