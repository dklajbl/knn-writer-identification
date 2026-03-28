import cv2
import numpy as np
from src.patchers.base_patcher import BasePatcher
from src.patchers.patcher_config import PatcherConfig


class RandomPatcher(BasePatcher):

    """
    Random patcher for text and document images.

    This patcher extracts exactly "patch_count" random patches of fixed size (patch_height, patch_width) from the input image.

    Behavior:
        1. First, it tries to collect random patches while skipping empty/padded ones.
        2. If the number of valid patches is still insufficient after a fixed number of attempts,
           it fills the remaining with random patches without checking whether they are empty.

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

    @staticmethod
    def _is_empty_or_padded_patch(
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

    def _normalize_patch_size(self, patch: np.ndarray) -> np.ndarray:

        """
        Resize patch to the output size if needed.

        Parameters:
            patch (np.ndarray): Input patch

        Returns:
            np.ndarray: Patch with shape (patch_height, patch_width, C).
        """

        if patch.shape[:2] != (self.patch_height, self.patch_width):
            patch = cv2.resize(
                patch,
                (self.patch_width, self.patch_height),
                interpolation=self.interpolation
            )

        return patch

    def _extract_random_patch(
        self,
        image: np.ndarray,
        crop_height: int,
        crop_width: int,
        max_y: int,
        max_x: int,
        skip_empty: bool = True,
    ) -> np.ndarray | None:

        """
        Extract one random patch from the image.

        Parameters:
            image (np.ndarray): Input image of shape (H, W, C).
            crop_height (int): Crop height before optional resizing.
            crop_width (int): Crop width before optional resizing.
            max_y (int): Maximum valid top-left y-coordinate.
            max_x (int): Maximum valid top-left x-coordinate.
            skip_empty (bool): If True, reject empty/padded patches and return None.

        Returns:
            np.ndarray | None: valid patch if extraction succeeds or return None if skip_empty=True and the patch is rejected
        """

        y = self.rng.integers(0, max_y + 1) if max_y > 0 else 0
        x = self.rng.integers(0, max_x + 1) if max_x > 0 else 0

        patch = image[y:y + crop_height, x:x + crop_width]

        if skip_empty and self._is_empty_or_padded_patch(patch):
            return None

        patch = self._normalize_patch_size(patch)

        return patch

    def extract_patches(self, image: np.ndarray) -> np.ndarray:

        """
        Extract random patches from the input image.

        Parameters:
            image (np.ndarray): Input image of shape (H, W, C).

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

        img_height, img_width, _ = image.shape

        # if the requested patch is larger than the image - crop what is available and resize to the target size
        crop_height = min(self.patch_height, img_height)
        crop_width = min(self.patch_width, img_width)

        max_y = max(0, img_height - crop_height)
        max_x = max(0, img_width - crop_width)

        patches = []  # final random patches

        # try to collect valid non-empty patches
        max_attempts = self.patch_count * 20
        attempts = 0

        while len(patches) < self.patch_count and attempts < max_attempts:
            attempts += 1

            patch = self._extract_random_patch(
                image=image,
                crop_height=crop_height,
                crop_width=crop_width,
                max_y=max_y,
                max_x=max_x,
                skip_empty=True,
            )
            if patch is not None:  # patch has not been rejected
                patches.append(patch)

        # if there are still not enough patches, fill the rest without checking whether the patches are empty
        while len(patches) < self.patch_count:
            patch = self._extract_random_patch(
                image=image,
                crop_height=crop_height,
                crop_width=crop_width,
                max_y=max_y,
                max_x=max_x,
                skip_empty=False,
            )
            patches.append(patch)

        return np.stack(patches, axis=0)
