import numpy as np
from abc import ABC, abstractmethod
from src.patchers.patcher_config import PatcherConfig


class BasePatcher(ABC):

    """
    Abstract base class for all patch extraction methods.

    Every concrete patcher receives a PatcherConfig and must implement `extract_patches` method, which returns a NumPy array of patches.
    """

    def __init__(self, config: PatcherConfig):

        """
        Initialize the patcher

        Parameters:
            config (PatcherConfig): configuration object describing how patch extraction should behave

        Raises:
            ValueError: if patch_count is not positive
        """

        if config.patch_count <= 0:
            raise ValueError("patch_count must be > 0")

        self.config = config
        self.patch_count = config.patch_count
        self.rng = np.random.default_rng(config.random_seed)

    @abstractmethod
    def extract_patches(self, image: np.ndarray) -> np.ndarray:

        """
        Extract patches from an image.

        Parameters:
            image (np.ndarray): input image as a NumPy array with shape (H, W, C).

        Returns:
            np.ndarray: extracted patches as a NumPy array with shape (N, patch_H, patch_W, C).
                N is the number of extracted patches, which may vary per image depending on the patcher.

        Raises:
            NotImplementedError: must be implemented by subclasses.
        """

        raise NotImplementedError
