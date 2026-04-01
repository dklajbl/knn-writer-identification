import math
import numpy as np
from src.patchers.base_patcher import BasePatcher
from src.patchers.patcher_config import PatcherConfig
from src.patchers.utils import normalize_patch_size


class GridPatcher(BasePatcher):

    """
    Adaptive rectangular grid patcher for text and document images

    Main properties:
        - works for both wide text-line images and more square document crops
        - chooses rows and columns automatically from the image aspect ratio
        - allows rows * cols >= patch_count
        - returns exactly "patch_count" patches
        - preserves reading order (top-to-bottom, left-to-right)
        - can resize patches to a common size for easy stacking
    """

    def __init__(self, config: PatcherConfig):

        """
        Initialize the grid patcher from the shared patcher configuration.

        Parameters:
            config (PatcherConfig): the patcher onfiguration object with grid-related options
        """

        super().__init__(config)
        self.max_rows = config.max_rows
        self.resize_patches = config.resize_patches
        self.interpolation = config.interpolation

    def _score_candidate(
        self,
        img_h: int,
        img_w: int,
        rows: int,
        cols: int,
    ) -> float:

        """
        Score a candidate grid shape. Lower scores are better.

        The score tries to balance three goals:
            1. The grid shape should roughly match the image aspect ratio
            2. The grid should not waste too many extra cells
            3. Wide text images should avoid too many rows

        Parameters:
            img_h (int): image height
            img_w (int): image width.
            rows (int): candidate number of grid rows
            cols (int): candidate number of grid columns

        Returns:
            float: a floating-point score. Lower is better.
        """

        img_aspect = img_w / max(img_h, 1)
        grid_aspect = cols / max(rows, 1)

        # main term: match the grid aspect ratio to the image aspect ratio.
        score = abs(math.log((grid_aspect + 1e-8) / (img_aspect + 1e-8)))

        # small penalty for unused cells if rows * cols > patch_count.
        wasted_cells = rows * cols - self.patch_count
        score += 0.12 * wasted_cells

        # text-aware bias: very wide images usually should not be split into too many rows
        if img_aspect >= 3.0:
            score += 0.25 * max(0, rows - 2)
        elif img_aspect >= 1.8:
            score += 0.18 * max(0, rows - 3)

        # for more square images, discourage forcing everything into one row
        if img_aspect <= 1.4 and rows == 1 and self.patch_count >= 4:
            score += 0.8

        return score

    def _choose_grid_shape(self, img_h: int, img_w: int) -> tuple[int, int]:

        """
        Choose the grid layout (rows, cols) for a given image.
        Find rows, cols with rows * cols >= patch_count and best fit to image aspect ratio.

        Parameters:
           img_h (int): image height
           img_w (int): image width

        Returns:
           tuple[int, int]: tuple of (rows, cols).

        Raises:
           ValueError: if no valid grid shape can be determined
        """

        if self.max_rows is not None:
            max_rows = min(self.max_rows, self.patch_count)
        else:
            # if no explicit limit is given, use a reasonable upper bound
            max_rows = min(
                self.patch_count,
                max(1, int(math.ceil(math.sqrt(self.patch_count))) + 2)
            )

        best: tuple[float, int, int] | None = None

        for rows in range(1, max_rows + 1):
            cols = math.ceil(self.patch_count / rows)
            score = self._score_candidate(img_h, img_w, rows, cols)

            candidate = (score, rows, cols)
            if best is None or candidate[0] < best[0]:
                best = candidate

        if best is None:
            raise ValueError("Could not determine grid shape")

        _, rows, cols = best
        return rows, cols

    def extract_patches(self, image: np.ndarray) -> np.ndarray:

        """
        Split the input image into an adaptive rectangular grid and return exactly "patch_count" patches.
        Patches are returned in row-major order: left-to-right within a row, then top-to-bottom across rows.
        If patch sizes differ slightly because of integer boundary rounding, patches can be resized to a common size before stacking.

        Parameters:
           image (np.ndarray): input image as a NumPy array with shape (H, W, C)

        Returns:
           np.ndarray: NumPy array of shape (patch_count, H_patch, W_patch, C)

        Raises:
           ValueError: if the input image is invalid or no patches can be extracted.
       """

        if image is None:
            raise ValueError("image is None")

        if image.ndim != 3:
            raise ValueError(f"Expected image shape (H, W, C), got {image.shape}")

        height, width, _ = image.shape
        rows, cols = self._choose_grid_shape(height, width)

        # compute integer grid boundaries that cover the whole image
        y_edges = np.linspace(0, height, rows + 1, dtype=int)
        x_edges = np.linspace(0, width, cols + 1, dtype=int)

        patches: list[np.ndarray] = []
        patch_shapes: list[tuple[int, int]] = []

        # extract patches in reading order
        for row_idx in range(rows):
            for col_idx in range(cols):

                y1, y2 = y_edges[row_idx], y_edges[row_idx + 1]
                x1, x2 = x_edges[col_idx], x_edges[col_idx + 1]

                patch = image[y1:y2, x1:x2]
                if patch.size == 0:
                    continue

                patches.append(patch)
                patch_shapes.append(patch.shape[:2])

                # stop once we have exactly the requested number of patches
                if len(patches) == self.patch_count:
                    break

            if len(patches) == self.patch_count:
                break

        if len(patches) == 0:
            raise ValueError("No patches could be extracted from the image.")

        if not self.resize_patches:
            # without resizing, all patches must already have identical shapes.

            unique_shapes = {patch.shape for patch in patches}

            if len(unique_shapes) != 1:
                raise ValueError(
                    "Patches have different shapes. "
                    "Enable resize_patches=True to stack them safely."
                )

            return np.stack(patches, axis=0)

        # use the median patch size as the target size - this is a stable choice when the edge patches differ by 1 pixel
        patch_heights, patch_widths = [shape[0] for shape in patch_shapes], [shape[1] for shape in patch_shapes]
        target_height, target_width = int(np.median(patch_heights)), int(np.median(patch_widths))

        normalized_patches = []

        for patch in patches:
            patch = normalize_patch_size(patch, target_height, target_width, self.interpolation)
            normalized_patches.append(patch)

        return np.stack(normalized_patches, axis=0)
