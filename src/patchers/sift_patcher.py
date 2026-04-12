import logging

import cv2
import numpy as np
from src.patchers.base_patcher import BasePatcher
from src.patchers.patcher_config import PatcherConfig
from src.patchers.random_patcher import RandomPatcher
from src.patchers.utils import is_empty_or_padded_patch, normalize_patch_size

logger = logging.getLogger(__name__)


class SIFTPatcher(BasePatcher):

    """
    SIFT-based patcher for text and document images.

    Behavior:
        1. detect SIFT keypoints
        2. sort them from strongest to weakest
        3. extract fixed-size patches centered at keypoints
        4. if there are too few valid patches, duplicate them in order
           from strongest to weakest until patch_count is reached
    """

    def __init__(self, config: PatcherConfig):

        """
        Initialize the SIFT patcher.

        Parameters:
            config (PatcherConfig): Patcher configuration.

        Raises:
            ValueError: If patch size is invalid.
        """

        super().__init__(config)

        self.patch_height = config.patch_height
        self.patch_width = config.patch_width
        self.interpolation = config.interpolation

        if self.patch_height <= 0 or self.patch_width <= 0:
            raise ValueError("patch_height and patch_width must be > 0")

        # cv2.SIFT cannot be pickled, so it is created lazily to support multiprocessing DataLoaders (which pickle the dataset and its patcher)
        self._sift = None

        # fallback patcher for when SIFT finds no keypoints or patch_count is not satisfied (to fill SIFT patches with random ones)
        self._random_fallback = RandomPatcher(config)

    @property
    def sift(self) -> cv2.SIFT:

        """
        Lazily create and return the OpenCV SIFT detector.

        The SIFT object is not created in __init__ because cv2.SIFT cannot be pickled,
        and PyTorch DataLoader with num_workers > 0 pickles the dataset (and its patcher)
        when spawning worker processes.

        Returns:
            cv2.SIFT: OpenCV SIFT feature detector.
        """

        if self._sift is None:
            self._sift = cv2.SIFT_create(nfeatures=self.patch_count)

        return self._sift

    def _extract_patch_around_keypoint(
        self,
        image: np.ndarray,
        keypoint: cv2.KeyPoint,
    ) -> np.ndarray | None:

        """
        Extract one fixed-size patch centered at a SIFT keypoint.

        Parameters:
            image (np.ndarray): Input image of shape (H, W, C).
            keypoint (cv2.KeyPoint): SIFT keypoint.

        Returns:
            np.ndarray | None: valid patch if extraction succeeds, None if the patch is empty/padded
        """

        img_height, img_width, _ = image.shape

        x_center = int(round(keypoint.pt[0]))
        y_center = int(round(keypoint.pt[1]))

        half_h = self.patch_height // 2
        half_w = self.patch_width // 2

        y1 = max(0, y_center - half_h)
        y2 = min(img_height, y_center + half_h)
        x1 = max(0, x_center - half_w)
        x2 = min(img_width, x_center + half_w)

        # crop around SIFT keypoint - with specific crop window size
        patch = image[y1:y2, x1:x2]

        if is_empty_or_padded_patch(patch):
            return None

        patch = normalize_patch_size(patch, self.patch_height, self.patch_width, self.interpolation)

        return patch

    @staticmethod
    def _duplicate_patches_to_count(
        patches: list[np.ndarray],
        target_count: int,
    ) -> list[np.ndarray]:

        """
        Duplicate patches in order from strongest to weakest until target_count is reached.

        Example:
            patches = [p1, p2, p3], target_count = 8
            result  = [p1, p2, p3, p1, p2, p3, p1, p2]

        Parameters:
            patches (list[np.ndarray]): Existing valid patches sorted from strongest to weakest.
            target_count (int): Desired number of output patches.

        Returns:
            list[np.ndarray]: Patch list of length target_count.

        Raises:
            ValueError: If the input patch list is empty.
        """

        if len(patches) == 0:
            raise ValueError("Cannot duplicate patches from an empty list.")

        result = []
        index = 0

        while len(result) < target_count:
            result.append(patches[index % len(patches)])
            index += 1

        return result

    def extract_patches(self, image: np.ndarray) -> np.ndarray:

        """
        Extract SIFT-guided fixed-size patches from the input image.

        Parameters:
            image (np.ndarray): Input image of shape (H, W, C).

        Returns:
            np.ndarray: Array of shape (patch_count, patch_height, patch_width, C).

        Raises:
            ValueError: If the input image is invalid or no valid SIFT patches are found.
        """

        if image is None:
            raise ValueError("image is None")

        if image.ndim != 3:
            raise ValueError(f"Expected image shape (H, W, C), got {image.shape}")

        # SPECIAL CASE: patch_count == 1 -> return full image (with expanded shape "patch_count"=1)
        if self.patch_count == 1:
            return np.expand_dims(image, axis=0)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        keypoints, _ = self.sift.detectAndCompute(gray, None)

        if keypoints is None:
            keypoints = []

        # sort from strongest to weakest
        keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)

        patches = []

        for keypoint in keypoints:
            if len(patches) >= self.patch_count:
                break

            patch = self._extract_patch_around_keypoint(image, keypoint)
            if patch is not None:
                patches.append(patch)

        if len(patches) == 0:
            logger.warning("SIFT found no valid patches, falling back to random patcher.")
            return self._random_fallback.extract_patches(image)

        if len(patches) < self.patch_count:
            # remaining patches (patch_count not satisfied) are filled with random ones

            remaining = self.patch_count - len(patches)
            original_count = self._random_fallback.patch_count

            try:
                if remaining == 1:  # if 1 patch is missing, make random patcher to extract 2 patches and choose just the 1st one - reason is when patch_count = 1, the random patcher returns original image
                    self._random_fallback.patch_count = 2
                    random_patches = self._random_fallback.extract_patches(image)
                    patches.append(random_patches[0])
                else:
                    self._random_fallback.patch_count = remaining
                    random_patches = self._random_fallback.extract_patches(image)
                    patches.extend(list(random_patches))
            finally:
                self._random_fallback.patch_count = original_count

        return np.stack(patches, axis=0)
