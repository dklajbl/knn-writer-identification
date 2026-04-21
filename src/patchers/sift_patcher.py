import lmdb
import pickle
import numpy as np
from src.patchers.base_patcher import BasePatcher
from src.patchers.patcher_config import PatcherConfig
from src.patchers.utils import normalize_patch_size


class SIFTPatcher(BasePatcher):

    """
    SIFT-based patcher that reads pre-computed keypoints from an LMDB.

    The keypoints LMDB is produced by `src/patchers/extract_keypoints.py`.
    Each record is keyed by the image name and holds a pickled dict with a (keypoint_count, 2) array of (y, x) rows.
    Detector keypoints are sorted strongest-first and padded with random keypoints so every image has exactly `keypoint_count` entries.

    At __getitem__ time this patcher:
        1. loads the record for the image key,
        2. validates that keypoint_count >= patch_count (hard error otherwise),
        3. takes the first `patch_count` keypoints,
        4. extracts a fixed-size patch centered at each (y, x).
    """

    def __init__(self, config: PatcherConfig):

        """
        Initialize the SIFT patcher.

        Parameters:
            config (PatcherConfig): Patcher configuration. `sift_keypoints_lmdb_path` must be set.

        Raises:
            ValueError: If patch size is invalid or the keypoints LMDB path is missing.
        """

        super().__init__(config)

        self.patch_height = config.patch_height
        self.patch_width = config.patch_width
        self.interpolation = config.interpolation

        if self.patch_height <= 0 or self.patch_width <= 0:
            raise ValueError("patch_height and patch_width must be > 0")

        if config.sift_keypoints_lmdb_path is None:
            raise ValueError(
                "SIFTPatcher requires `sift_keypoints_lmdb_path` in PatcherConfig. "
                "Run `python -m src.patchers.extract_keypoints ...` first to build the LMDB."
            )

        self.keypoints_lmdb_path = config.sift_keypoints_lmdb_path

        # LMDB objects are opened lazily so worker processes spawned by DataLoader get their own handle
        self._env: lmdb.Environment | None = None
        self._txn: lmdb.Transaction | None = None

    def _ensure_lmdb_open(self) -> None:

        """
        Open the keypoints LMDB on first use. Safe under DataLoader multiprocessing because each worker
        process pickles its patcher (without the env) and opens a fresh read-only handle here.
        """

        if self._txn is None:
            self._env = lmdb.open(
                self.keypoints_lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
            )
            self._txn = self._env.begin(write=False)

    def _extract_patch_at(
        self,
        image: np.ndarray,
        y_center: int,
        x_center: int,
    ) -> np.ndarray:

        """
        Extract one fixed-size patch centered at (y_center, x_center), clipping at image borders and resizing to the target (patch_height, patch_width) when the clip produced a smaller crop.

        Parameters:
            image (np.ndarray): Input image of shape (H, W, C).
            y_center (int): Row coordinate of the keypoint.
            x_center (int): Column coordinate of the keypoint.

        Returns:
            np.ndarray: Patch of shape (patch_height, patch_width, C).
        """

        img_height, img_width, _ = image.shape

        half_h = self.patch_height // 2
        half_w = self.patch_width // 2

        y1 = max(0, y_center - half_h)
        y2 = min(img_height, y_center + half_h)
        x1 = max(0, x_center - half_w)
        x2 = min(img_width, x_center + half_w)

        patch = image[y1:y2, x1:x2]
        patch = normalize_patch_size(patch, self.patch_height, self.patch_width, self.interpolation)

        return patch

    def extract_patches(self, image: np.ndarray, key: str | None = None) -> np.ndarray:

        """
        Extract fixed-size patches centered at the pre-computed SIFT keypoints for `key`.

        Parameters:
            image (np.ndarray): Input image of shape (H, W, C).
            key (str): Image name used to look up keypoints in the LMDB.

        Returns:
            np.ndarray: Array of shape (patch_count, patch_height, patch_width, C).

        Raises:
            ValueError: If the input image is invalid, `key` is missing, or the LMDB stores fewer keypoints per image than the requested `patch_count`.
            KeyError: If `key` is not present in the keypoints LMDB.
        """

        if image is None:
            raise ValueError("image is None")

        if image.ndim != 3:
            raise ValueError(f"Expected image shape (H, W, C), got {image.shape}")

        if key is None:
            raise ValueError("SIFTPatcher.extract_patches requires `key` (the image name) to look up keypoints.")

        # SPECIAL CASE: patch_count == 1 -> return full image (with expanded shape "patch_count"=1)
        if self.patch_count == 1:
            return np.expand_dims(image, axis=0)

        self._ensure_lmdb_open()
        raw = self._txn.get(key.encode())

        if raw is None:
            raise KeyError(
                f"Image key {key!r} not found in SIFT keypoints LMDB '{self.keypoints_lmdb_path}'. "
                f"Make sure extract_keypoints.py was run over the same image LMDB this split references."
            )

        record = pickle.loads(raw)
        keypoints = record["keypoints"]  # (keypoint_count, 2), rows = (y, x)

        stored_count = int(keypoints.shape[0])

        if stored_count < self.patch_count:
            raise ValueError(
                f"SIFT keypoints LMDB stores only {stored_count} keypoints per image, but patch_count={self.patch_count}."
                f"Re-run extract_keypoints.py with --keypoint-count >= {self.patch_count}."
            )

        selected = keypoints[: self.patch_count]

        patches = [
            self._extract_patch_at(image, int(y), int(x))
            for y, x in selected
        ]

        return np.stack(patches, axis=0)
