import cv2
from typing import Literal
from dataclasses import dataclass


# all supported patching methods
PATCH_METHODS = ["grid", "random", "algorithmic"]
PatchMethodLiteral = Literal["grid", "random", "algorithmic"]  # a literal for variable types


@dataclass(frozen=True)
class PatcherConfig:

    """
    Configuration object for image patch extraction.

    Attributes:
        method (default = "grid"): what patching method to use
            - "grid": split the image into an adaptive rectangular grid
            - "random":
            - "algorithmic":
        patch_count (default = 16): number of patches that should be extracted from full image
        random_seed (default = None): optional rng seed for reproducible random-based patchers
        max_rows (default = 4):
            - using "grid" method (for now anyway):
                Optional upper bound for the number of rows in the grid.
                This is useful for text images because too many rows can break the reading structure, especially for wide line images
        resize_patches (default = True): if True, patches are resized to a common shape before stacking
        interpolation (default = cv2.INTER_LINEAR): OpenCV interpolation flag used when resizing patches
        patch_height (default = 32): height of each patch (only used for random or algorithmic patching)
        patch_width (default = 32): width of each patch (only used for random or algorithmic patching)
    """

    method: PatchMethodLiteral = "grid"
    patch_count: int = 16
    random_seed: int | None = None

    # options used by the "grid" patcher
    max_rows: int | None = 4
    resize_patches: bool = True
    interpolation: int = cv2.INTER_LINEAR

    # random / algorithmic patch size
    patch_height: int = 20
    patch_width: int = 20
