import numpy as np
from src.patchers.base_patcher import BasePatcher
from src.patchers.patcher_config import PatcherConfig

class SinglePatcher(BasePatcher):

    def __init__(self, config: PatcherConfig):
        super().__init__(config)

        self.patch_height = config.patch_height
        self.patch_width = config.patch_width

    def extract_patches(self, image: np.ndarray, key: str | None = None) -> np.ndarray:

        if image is None:
            raise ValueError("image is None")

        if image.ndim != 3:
            raise ValueError(f"Expected image shape (H, W, C), got {image.shape}")

        # get current image shape
        H, W, C = image.shape

        # get target image shape
        target_h, target_w = self.patch_height, self.patch_width

        # adjust height to target height
        if H >= target_h:
            # crop height (top and bottom equally), if the target is smaller
            top = (H - target_h) // 2
            image = image[top:top + target_h, :, :]
        else:
            # zero pad height (top and bottom equally), if the target is larger
            pad_top = (target_h - H) // 2
            pad_bottom = target_h - H - pad_top
            image = np.pad(
                image,
                ((pad_top, pad_bottom), (0, 0), (0, 0)),
                mode="constant"
            )

        # adjust width to target width
        if W >= target_w:
            # crop width (left and right equally), if the target is smaller
            left = (W - target_w) // 2
            image = image[:, left:left + target_w, :]
        else:
            # pad width (left and right equally), if the target is larger
            pad_left = (target_w - W) // 2
            pad_right = target_w - W - pad_left
            image = np.pad(
                image,
                ((0, 0), (pad_left, pad_right), (0, 0)),
                mode="constant"
            )

        # final shape: (H', W', C)
        patch = image

        # add batch dimension
        # (H', W', C) -> (1, H', W', C)
        patch = patch[np.newaxis]

        return patch
