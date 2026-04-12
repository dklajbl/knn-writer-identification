import numpy as np
from src.patchers.base_patcher import BasePatcher
from src.patchers.patcher_config import PatcherConfig
from src.patchers.utils import normalize_patch_size


class GridPatcher(BasePatcher):

    """
    Fixed-size sliding window patcher for text and document images.

    Extracts patches of a fixed size (patch_height x patch_width) in reading order:
        left-to-right within each row, top-to-bottom across rows.

    Unlike the random and SIFT patchers, the grid patcher does not use a fixed patch_count.
    Instead, it extracts as many patches as fit within the image. Partial edge patches (at the right or bottom border)
    are included and resized to the target dimensions if their remaining extent is at least min_partial_ratio of the full patch dimension.
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
            np.ndarray: extracted patches with shape (N, patch_height, patch_width, C),
                where N is the number of patches that fit in this image.

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

        img_h, img_w, _ = image.shape

        # minimum pixel extent for an edge remainder to be kept as a partial patch
        min_h = self.patch_height * self.min_partial_ratio
        min_w = self.patch_width * self.min_partial_ratio

        patches: list[np.ndarray] = []

        # slide top-to-bottom in steps of patch_height
        y = 0
        while y < img_h:
            remaining_h = img_h - y

            if remaining_h < min_h:
                # remaining strip is too thin to be useful - stop iterating rows
                break

            actual_h = min(self.patch_height, remaining_h)

            # slide left-to-right in steps of patch_width
            x = 0
            while x < img_w:
                remaining_w = img_w - x

                if remaining_w < min_w:
                    # remaining strip is too narrow - skip to next row
                    break

                actual_w = min(self.patch_width, remaining_w)

                # extract the patch (full or partial edge patch)
                patch = image[y:y + actual_h, x:x + actual_w]

                # resize to the target dimensions (no-op for full patches that already match)
                patch = normalize_patch_size(patch, self.patch_height, self.patch_width, self.interpolation)
                patches.append(patch)

                x += self.patch_width

            y += self.patch_height

        if len(patches) == 0:
            raise ValueError("No patches could be extracted from the image.")

        return np.stack(patches, axis=0)
