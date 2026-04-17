import numpy as np
from src.patchers.base_patcher import BasePatcher
from src.patchers.patcher_config import PatcherConfig
from src.patchers.utils import normalize_patch_size


class GridPatcher(BasePatcher):

    """
    Fixed-size sliding window patcher for text and document images.

    Extracts patches of a fixed size (patch_height x patch_width) in reading order: left-to-right within each row, top-to-bottom across rows.

    Unlike the random and SIFT patchers, the grid patcher does not use a fixed patch_count.
    Instead, it extracts as many patches as fit within the image.

    Partial edge patches (at the right or bottom border) are included and resized to the target dimensions if their remaining extent is at least min_partial_ratio of the full patch dimension.
    Strips thinner than that threshold are skipped to avoid heavy interpolation artifacts.

    The variable patch count per image is handled downstream by a custom collate function that pads batches to the maximum count and produces a padding mask.
    """

    def __init__(self, config: PatcherConfig):

        """
        Initialize the grid patcher from the shared patcher configuration.

        Parameters:
            config (PatcherConfig): the patcher configuration object with grid-related options
        """

        super().__init__(config)
        self.patch_height = config.patch_height
        self.patch_width = config.patch_width
        self.interpolation = config.interpolation
        self.min_partial_ratio = config.min_partial_ratio

    def extract_patches(self, image: np.ndarray) -> np.ndarray:

        """
        Extract fixed-size patches from the image in reading order (left-to-right, top-to-bottom).

        Full patches are taken at exact (patch_height x patch_width). At image edges, partial patches
        are included if their remaining extent is >= min_partial_ratio of the patch dimension, and
        resized to the full patch size. Thinner remainders are skipped.

        Parameters:
            image (np.ndarray): input image as a NumPy array with shape (H, W, C).

        Returns:
            np.ndarray: extracted patches with shape (N, patch_height, patch_width, C), where N is the number of patches that fit in this image.

        Raises:
            ValueError: if the input image is None, has wrong dimensions, or yields no patches.
        """

        if image is None:
            raise ValueError("image is None")

        if image.ndim != 3:
            raise ValueError(f"Expected image shape (H, W, C), got {image.shape}")

        # SPECIAL CASE: patch_count == 1 -> return full image (with expanded shape "patch_count"=1)
        if self.patch_count == 1:
            return np.expand_dims(image, axis=0)

        img_height, img_width, channels = image.shape
        patch_height, patch_width = self.patch_height, self.patch_width

        # minimum pixel extent for an edge remainder to be kept as a partial patch
        min_partial_height = patch_height * self.min_partial_ratio
        min_partial_width = patch_width * self.min_partial_ratio

        # how many full-size patches fit in each direction
        num_full_rows = img_height // patch_height
        num_full_cols = img_width // patch_width
        remaining_height = img_height - num_full_rows * patch_height
        remaining_width = img_width - num_full_cols * patch_width

        has_partial_bottom = remaining_height >= min_partial_height
        has_partial_right = remaining_width >= min_partial_width

        # extract all full interior patches using reshape
        full_grid = None
        if num_full_rows > 0 and num_full_cols > 0:
            full_region = image[:num_full_rows * patch_height, :num_full_cols * patch_width, :].copy()
            full_grid = full_region.reshape(
                num_full_rows, patch_height, num_full_cols, patch_width, channels
            ).transpose(0, 2, 1, 3, 4)  # shape: (num_full_rows, num_full_cols, patch_height, patch_width, channels)

        patches: list[np.ndarray] = []
        total_rows = num_full_rows + (1 if has_partial_bottom else 0)

        for row in range(total_rows):
            is_partial_row = (row >= num_full_rows)

            if not is_partial_row and full_grid is not None:
                # bulk-append all full patches in this row (no resize needed)
                patches.append(full_grid[row])  # (num_full_cols, patch_height, patch_width, channels)

            elif is_partial_row:
                # bottom partial row: extract and resize each column patch

                top = num_full_rows * patch_height
                for col in range(num_full_cols):
                    left = col * patch_width
                    patch = image[top:, left:left + patch_width, :]
                    patches.append(normalize_patch_size(patch, patch_height, patch_width, self.interpolation)[np.newaxis])

            # right edge partial patch for this row (resize needed)
            if has_partial_right:
                left = num_full_cols * patch_width

                if is_partial_row:
                    top = num_full_rows * patch_height
                    patch = image[top:, left:, :]
                else:
                    top = row * patch_height
                    patch = image[top:top + patch_height, left:, :]

                patches.append(normalize_patch_size(patch, patch_height, patch_width, self.interpolation)[np.newaxis])

        if len(patches) == 0:
            raise ValueError("No patches could be extracted from the image.")

        return np.concatenate(patches, axis=0)
