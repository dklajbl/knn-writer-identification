import lmdb
import random
import logging
from collections import defaultdict

import cv2
import numpy as np
np.bool = bool

import imgaug.augmenters as iaa
import torch

from src.patchers.patcher_config import PatcherConfig
from src.patchers.make_patcher import make_patcher


logger = logging.getLogger(__name__)


class IdDataset(torch.utils.data.Dataset):

    """
    PyTorch Dataset class for loading line images from an LMDB database.

    Each sample is a single line image identified by (cluster_id, image_name).
    Images are loaded from LMDB, optionally augmented, patched, and converted to tensors.
    """

    def __init__(
        self,
        file_name: str,
        lmdb_path: str,
        augment: bool = True,
        restrict_data: bool = False,
        test: bool = False,
        patcher_config: PatcherConfig | None = None,
    ):
        """
        Initialize the dataset

        Parameters:
            file_name (str): Path to the file containing image names and IDs.
            lmdb_path (str): Path to the LMDB database
            augment (bool): Whether to use image augmentation or not
            restrict_data (bool): Whether to restrict dataset size
            test (bool): Whether the dataset is used in test mode
            patcher_config (PatcherConfig): Patcher configuration

        Raises:
            ValueError: if patcher config is not defined
        """

        super().__init__()

        if patcher_config is None:
            raise ValueError("patcher_config must be provided. Images are always patched.")

        self.file_name = file_name
        self.lmdb_path = lmdb_path
        self.augment = augment
        self.restrict_data = restrict_data
        self.test = test

        self.patcher = make_patcher(patcher_config)

        # LMDB objects are initialized lazily, this prevents problems with multiprocessing DataLoaders
        self.env = None
        self.txn = None

        self.aug = None  # augmentation pipeline

        # list of tuples (cluster_id, image_name)
        self.lines = []

        # defaultdict: cluster_id -> list of image names
        self.id_lines = defaultdict(list)

        self._load_lines()

        if self.augment:  # oversample smaller classes to reduce imbalance
            self.uniformize_data_distribution()

        if self.restrict_data:  # optionally reduce dataset size
            self.restrict_data_distribution()

    def _load_lines(self) -> None:

        """
        Load lines from the input file and group them by ID

        The index file contains:
            image_name cluster_id

        Example:
            img_001.jpg 3
            img_002.jpg 3
            img_003.jpg 7
        """

        with open(self.file_name, 'r', encoding="utf-8") as file:
            raw_lines = file.readlines()

        parsed_lines = [line.split()[:2] for line in raw_lines]

        self.lines = [
            (int(parts[1]), parts[0])
            for parts in parsed_lines
            if len(parts) == 2
        ]

        for cluster_id, image_name in self.lines:
            self.id_lines[cluster_id].append(image_name)

    def _ensure_lmdb_open(self) -> None:

        """
        Open LMDB database if it is not already opened.
        LMDB should be opened lazily because PyTorch DataLoader may create multiple worker processes.
        """

        if self.txn is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
            )
            self.txn = self.env.begin(write=False)

        # create augmentation pipeline only once
        if self.augment and self.aug is None:
            self.aug = self._build_augmenter()

    @staticmethod
    def _build_augmenter() -> iaa.SomeOf:

        """
        Create augmentation pipeline using imgaug.

        Returns:
            iaa.SomeOf: Random augmentation pipeline
        """

        return iaa.SomeOf(
            n=(1, 4),  # randomly apply between 1 and 4 augmentations
            children=[
                iaa.convolutional.DirectedEdgeDetect(alpha=(0.05, 0.2), direction=(0.0, 1.0)),
                iaa.convolutional.EdgeDetect(alpha=(0.05, 0.15)),
                iaa.convolutional.Emboss(alpha=(0.05, 0.2), strength=(0.2, 0.7)),
                iaa.convolutional.Sharpen(alpha=(0.05, 0.2), lightness=(0.8, 1.2)),
                iaa.contrast.AllChannelsCLAHE(clip_limit=(0.1, 8), tile_grid_size_px=(3, 12), tile_grid_size_px_min=3),
                iaa.contrast.CLAHE(clip_limit=(0.1, 8), tile_grid_size_px=(3, 12), tile_grid_size_px_min=3),
                iaa.contrast.GammaContrast(gamma=(0.6, 1.8)),
                iaa.contrast.LogContrast(gain=(0.6, 1.4)),
                iaa.BlendAlpha((0.2, 0.7), iaa.contrast.AllChannelsHistogramEqualization()),
                iaa.BlendAlpha((0.2, 0.7), iaa.contrast.HistogramEqualization()),
                iaa.blur.GaussianBlur(sigma=(0.0, 2.5)),
                iaa.pillike.EnhanceContrast(factor=(0.5, 1.5)),
                iaa.pillike.EnhanceBrightness(factor=(0.5, 1.5)),
                iaa.pillike.EnhanceSharpness(factor=(0.5, 1.5)),
                iaa.pillike.FilterEdgeEnhance(),
                iaa.pillike.FilterSharpen(),
                iaa.pillike.FilterDetail()
            ],
        )

    def restrict_data_distribution(
        self,
        max_id: int = -1,
        max_id_size: int = 5000,
        min_id_size: int = 0
    ) -> None:

        """
        Restrict dataset size to avoid extremely large classes.

        Parameters:
            max_id (int): Maximum number of IDs kept.
            max_id_size (int): Maximum samples per ID.
            min_id_size (int): Minimum samples required for an ID.
        """

        logger.info(f"{self.file_name} -> restrict data distribution from {len(self.id_lines)} ids, {len(self.lines)} lines",)

        new_id_lines = defaultdict(list)

        for cluster_id in self.id_lines.keys():

            # ignore IDs that do not have enough samples
            if len(self.id_lines[cluster_id]) >= min_id_size:
                shuffled_names = self.id_lines[cluster_id][:]
                random.shuffle(shuffled_names)

                # keep at most max_id_size samples from this ID.
                new_id_lines[cluster_id] = shuffled_names[:max_id_size]

        items = list(new_id_lines.items())
        random.shuffle(items)

        if max_id != -1:
            # keep only the selected number of IDs
            items = items[:max_id]

        self.id_lines = dict(items)

        # rebuild flat sample list after restriction
        self.lines = []
        for cluster_id in self.id_lines:
            self.lines += [(cluster_id, name) for name in self.id_lines[cluster_id]]

        logger.info(f"{self.file_name} -> max_id:{max_id}, max_id_size:{max_id_size}, min_id_size:{min_id_size}, restricted to {len(self.id_lines)} ids, {len(self.lines)} lines")

    def uniformize_data_distribution(self, p: float = 0.65) -> None:

        """
        Oversample smaller classes to reduce class imbalance

        Parameters:
            p (float): rebalancing strength
        """

        ids = sorted([key for key in self.id_lines])
        counts = np.asarray([len(self.id_lines[key]) for key in ids])
        max_count = np.max(counts)

        # smaller classes get increased, but not fully to max_count.
        new_counts = (counts ** p / max_count ** p) * max_count

        logger.info(f"{self.file_name} -> uniformize data distribution from {len(self.lines)} lines",)

        for cluster_id, old_count, new_count in zip(ids, counts, new_counts):
            new_count = int(new_count)

            # add repeated samples from the same ID until target count is reached.
            for _ in range(new_count - old_count):
                self.lines.append((cluster_id, random.choice(self.id_lines[cluster_id])))

        logger.info(f"{self.file_name} -> p: {p}, max_count:{max_count}, uniformize data distribution to {len(self.lines)} lines",)

    def id_count(self) -> int:

        """
        Return the number of IDs

        Returns:
            int: Number of IDs.
        """

        return np.max(list(self.id_lines)) + 1

    def __len__(self) -> int:

        """
        Return number of samples in dataset.

        Returns:
            int: Dataset size.
        """

        return len(self.lines)

    def _read_line(self, name: str) -> np.ndarray | None:

        """
        Load image from LMDB and convert it to numpy array.

        Parameters:
            name (str): key of the image inside LMDB

        Returns:
            np.ndarray | None: Loaded image of shape (H, W, C) or None if loading fails.
        """

        self._ensure_lmdb_open()

        data = self.txn.get(name.encode())

        if data is None:
            logger.warning(f"Unable to load image '{name}' specified in '{self.file_name}' from DB '{self.lmdb_path}'.")
            return None

        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        if image is None:
            logger.warning(f"Unable to decode image '{name}'.")
            return None

        # expand to (H, W, 1) so patchers receive 3D input
        image = np.expand_dims(image, axis=-1)

        return image

    def get_single_id_lines(self, idx: int, line_count: int = 32) -> list[np.ndarray] | None:

        """
        Load a random subset of images for one ID.

        Parameters:
            idx (int): Cluster ID.
            line_count (int): Maximum number of images to load

        Returns:
            list[np.ndarray] | None: List of images or None.
        """

        if idx not in self.id_lines:
            return None

        lines = np.random.choice(
            self.id_lines[idx],
            size=min(line_count, len(self.id_lines[idx])),
            replace=False
        )

        images = [self._read_line(line_name) for line_name in lines]
        images = [img for img in images if img is not None]

        if not images:
            return None

        return images

    def get_single_id_all_lines(self, idx: int) -> list[np.ndarray]:

        """
        Load all images belonging to one ID

        Parameters:
            idx (int): Cluster ID.

        Returns:
            list[np.ndarray]: List of images.
        """

        images = [self._read_line(line_name) for line_name in self.id_lines[idx]]
        images = [img for img in images if img is not None]

        if len(images) == 0:
            raise ValueError(f"No images could be loaded for id: {idx}")

        return images

    def get_original_image1(self, idx: int) -> torch.Tensor:

        """
        Get an original image in full, without it being patched (augmentation is still applied).

        Parameters:
            idx (int): Dataset index.

        Returns:
            torch.Tensor: Original image tensor of shape (C, H, W).
        """

        image = self._read_line(self.lines[idx][1])

        if self.aug is not None:
            image = self.aug(images=[image])[0]

        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)

        return image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:

        """
        Return a positive pair of images from the same ID.

        Parameters:
            idx (int): Dataset index.

        Returns:
            tuple[torch.Tensor, torch.Tensor, int]: (image_1, image_2, line_cluster_id)
        """

        image_1 = self._read_line(self.lines[idx][1])
        line_cluster_id = self.lines[idx][0]

        if image_1 is None:
            raise RuntimeError(f"Failed to load image '{self.lines[idx][1]}' (index {idx}) from LMDB '{self.lmdb_path}'.")

        # pick another sample from the same class to create a positive pair.
        line_name = np.random.choice(self.id_lines[line_cluster_id])
        image_2 = self._read_line(line_name)

        if image_2 is None:
            raise RuntimeError(
                f"Failed to load image '{line_name}' (positive pair for index {idx}) from LMDB '{self.lmdb_path}'."
            )

        if self.aug is not None:
            image_1, image_2 = self.aug(images=[image_1, image_2])

        # patch both images
        image_1 = self.patcher.extract_patches(image_1)
        image_2 = self.patcher.extract_patches(image_2)

        # convert numpy to tensor, scaled to [0, 1]
        image_1 = torch.from_numpy(image_1.transpose(0, 3, 1, 2).copy()).float().div_(255.0)
        image_2 = torch.from_numpy(image_2.transpose(0, 3, 1, 2).copy()).float().div_(255.0)

        return image_1, image_2, line_cluster_id

    def get_characters(self) -> list:

        """
        Return unique characters appearing in stored image names.

        Returns:
            list: Unique characters.
        """

        return list(set("".join([item[1] for item in self.lines])))
