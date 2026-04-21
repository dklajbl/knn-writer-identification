import cv2
from typing import Literal
from dataclasses import dataclass


# all supported patching methods
PATCH_METHODS = ["grid", "random", "sift"]
PatchMethodLiteral = Literal["grid", "random", "sift"]  # a literal for variable types


@dataclass(frozen=True)
class PatcherConfig:

    """
    Configuration object for image patch extraction.

    Attributes:
        method (default = "grid"): what patching method to use
            - "grid": extract fixed-size patches in reading order (left-to-right, top-to-bottom)
            - "random": extract patches at random positions
            - "sift": extract patches centered at SIFT keypoints
        patch_count (default = 16): number of patches to extract (used by random and sift patchers; ignored by grid patcher which extracts as many as fit)
        random_seed (default = None): optional rng seed for reproducible random-based patchers
        interpolation (default = cv2.INTER_LINEAR): OpenCV interpolation flag used when resizing patches
        patch_height (default = 20): height of each extracted patch in pixels (used by all patchers)
        patch_width (default = 20): width of each extracted patch in pixels (used by all patchers)
        min_partial_ratio (default = 0.3): minimum fraction of patch dimension that an edge remainder (remaining width) must have to be included as a partial patch (grid patcher only).
            Remainders below this threshold are skipped because resizing such a thin strip introduces heavy interpolation artifacts.
            For example, with patch_width=32 and min_partial_ratio=0.3, edge strips narrower than ~10px are skipped.
        sift_keypoints_lmdb_path (default = None): path to an LMDB produced by `src/patchers/extract_keypoints.py`.
            Required when method == "sift". Each record is keyed by the image name and stores a pre-computed array of (y, x) coordinates, so the SIFT patcher can look up keypoints instead of re-running cv2.SIFT.detect for every sample.
    """

    method: PatchMethodLiteral = "grid"
    patch_count: int = 16
    random_seed: int | None = None

    interpolation: int = cv2.INTER_LINEAR

    # patch dimensions
    patch_height: int = 20
    patch_width: int = 20

    # grid patcher: minimum edge remainder ratio to keep a partial patch
    min_partial_ratio: float = 0.3

    # sift patcher: path to a pre-computed keypoints LMDB (see extract_keypoints.py)
    sift_keypoints_lmdb_path: str | None = None
