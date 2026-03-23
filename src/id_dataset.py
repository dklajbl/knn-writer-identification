import re
import uuid
import lmdb
import random
import logging
from collections import defaultdict
from typing import Callable

import cv2
import numpy as np
import imgaug.augmenters as iaa
import torch

from patchers.patcher_config import PatcherConfig
from patchers.make_patcher import make_patcher


logger = logging.getLogger(__name__)


class IdDataset(torch.utils.data.Dataset):

    """
    PyTorch Dataset class for loading image lines or page segments from an LMDB database

     The dataset supports two modes:
        - page=False: each sample is a single line image
        - page=True: each sample belongs to a page and may be split into multiple windows
    """

    def __init__(
        self,
        file_name: str,
        lmdb_path: str,
        width: int = 320,
        transform: Callable | None = None,
        augment: bool = True,
        restrict_data: bool = False,
        test: bool = False,
        page: bool = False,
        patcher_config: PatcherConfig | None = None,
    ):
        """
        Initialize the dataset

        Parameters:
            file_name (str): Path to the file containing image names and IDs.
            lmdb_path (str): Path to the LMDB database
            width (int): Target image width
            transform (callable): transform applied to images (resize, toTensor, ...)
            augment (bool): Whether to use image augmentation or not
            restrict_data (bool): Whether to restrict dataset size
            test (bool): Whether the dataset is used in test mode
            page (bool): Whether dataset works in page mode
            patcher_config (PatcherConfig): Patcher configuration

        Raises:
            ValueError: if patcher config is not defined
        """

        super().__init__()

        if patcher_config is None:
            raise ValueError("patcher_config must be provided. Images are always patched.")

        self.file_name = file_name
        self.lmdb_path = lmdb_path
        self.width = width
        self.transform = transform
        self.augment = augment
        self.restrict_data = restrict_data
        self.test = test
        self.page = page

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

        The index file usually contains:
            image_name cluster_id

        Example:
            img_001.jpg 3
            img_002.jpg 3
            img_003.jpg 7
        """

        with open(self.file_name, 'r', encoding="utf-8") as file:
            raw_lines = file.readlines()

        if not self.page:
            # in normal mode, each line contains image_name and numeric cluster_id.

            parsed_lines = [line.split()[:2] for line in raw_lines]

            self.lines = [
                (int(parts[1]), parts[0])
                for parts in parsed_lines
                if len(parts) == 2
            ]

            for cluster_id, image_name in self.lines:
                self.id_lines[cluster_id].append(image_name)

        else:
            # in page mode, the first token is a line_id.
            # several line_ids can belong to the same page_id.

            line_ids = [line.split()[0].strip() for line in raw_lines if line.split()]
            self.lines = [
                (self.convert_line_id_to_page_id(line_id), line_id)
                for line_id in line_ids
            ]

            for page_id, line_id in self.lines:
                self.id_lines[page_id].append(line_id)

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
                iaa.color.AddToHue(value=(-64, 64)),
                iaa.color.AddToBrightness(add=(-40, 40)),
                iaa.color.AddToSaturation(value=(-64, 64)),
                iaa.color.Grayscale(),
                iaa.color.Grayscale(),
                iaa.color.MultiplyBrightness(mul=(0.8, 1.2)),
                iaa.color.MultiplyHue(mul=(-0.7, 0.7)),
                iaa.color.MultiplySaturation(mul=(0.0, 2.0)),
                iaa.color.Posterize(nb_bits=(2, 8)),
                iaa.contrast.AllChannelsCLAHE(clip_limit=(0.1, 8), tile_grid_size_px=(3, 12), tile_grid_size_px_min=3),
                iaa.contrast.CLAHE(clip_limit=(0.1, 8), tile_grid_size_px=(3, 12), tile_grid_size_px_min=3),
                iaa.contrast.GammaContrast(gamma=(0.6, 1.8)),
                iaa.contrast.LogContrast(gain=(0.6, 1.4)),
                iaa.BlendAlpha((0.2, 0.7), iaa.contrast.AllChannelsHistogramEqualization()),
                iaa.BlendAlpha((0.2, 0.7), iaa.contrast.HistogramEqualization()),
                iaa.blur.BilateralBlur(d=(1, 7), sigma_color=(10, 250), sigma_space=(10, 250)),
                iaa.blur.GaussianBlur(sigma=(0.0, 2.5)),
                iaa.pillike.Solarize(p=1.0, threshold=128),
                iaa.pillike.EnhanceColor(factor=(0.5, 1.5)),
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
                self.lines.append((cluster_id, np.random.choice(self.id_lines[cluster_id])))

        logger.info(f"{self.file_name} -> p: {p}, max_count:{max_count}, uniformize data distribution to {len(self.lines)} lines",)

    def id_count(self) -> int:

        """
        Return the number of IDs

        Returns:
            int: Number of IDs.
        """

        if self.page:
            return len(self.id_lines.keys())

        # in normal mode, labels are expected to be integer IDs
        return np.max(list(self.id_lines)) + 1

    def __len__(self) -> int:

        """
        Return number of samples in dataset.

        Returns:
            int: Dataset size.
        """

        return len(self.lines)

    def _read_line(self, name: str) -> np.ndarray | list | None:

        """
        Load image from LMDB and convert it to numpy array.

        In non-page mode: returns a single image
        In page mode: returns a list of overlapping windows

        Parameters:
            name (str): key of the image inside LMDB

        Returns:
            numpy.ndarray | list | None: Loaded image or list of page windows.
        """

        self._ensure_lmdb_open()

        data = self.txn.get(name.encode())

        if data is None:
            logger.warning(f"Unable to load image '{name}' specified in '{self.file_name}' from DB '{self.lmdb_path}'.")
            return None

        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            logger.warning(f"Unable to decode image '{name}'.")
            return None

        if not self.page:
            return self._prepare_single_image(image)

        return self._prepare_page_images(image)

    def _prepare_single_image(self, image: np.ndarray) -> np.ndarray:

        """
        Crop or pad one image to fixed width.

        Parameters:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Processed image.
        """
        if image.shape[1] > self.width:

            if self.test:
                # in test mode use center crop so output is deterministic
                pos = image.shape[1] // 2 - self.width // 2

            else:
                # in training use random crop as mild augmentation.
                pos = np.random.randint(0, image.shape[1] - self.width + 1)

            image = image[:, pos:][:, :self.width]

        elif image.shape[1] < self.width:
            # pad narrower images so all samples have equal width
            padded = np.zeros((image.shape[0], self.width, 3), dtype=np.uint8)
            padded[:, :image.shape[1]] = image
            image = padded

        return image

    def _prepare_page_images(self, image: np.ndarray) -> list:

        """
        Split a wide page image into overlapping windows

        Parameters:
            image (numpy.ndarray): Input page image.

        Returns:
            list: List of fixed-width image windows.
        """

        if image.shape[1] > self.width:
            images = []
            pos = 0

            while True:
                new_image = image[:, pos:pos + self.width]
                images.append(new_image)

                if pos == image.shape[1] - self.width:
                    break

                # move with overlap so page content is not skipped.
                pos = min(pos + 50, image.shape[1] - self.width)

        elif image.shape[1] < self.width:
            padded = np.zeros((image.shape[0], self.width, 3), dtype=np.uint8)
            padded[:, :image.shape[1]] = image
            images = [padded]

        else:
            images = [image]

        return images

    def get_single_id_lines(self, idx: int | str, line_count: int = 32) -> np.ndarray | None:

        """
        Load a random subset of images for one ID.

        Parameters:
            idx (int | str): Cluster ID or page ID.
            line_count (int): Maximum number of images to load

        Returns:
            numpy.ndarray | None: stacked images or None.
        """

        if idx not in self.id_lines:
            return None

        lines = np.random.choice(
            self.id_lines[idx],
            size=min(line_count, len(self.id_lines[idx])),
            replace=False
        )

        images = [self._read_line(line_name) for line_name in lines]
        images = np.stack(images, axis=0)

        return images

    def get_single_id_all_lines(self, idx: int | str) -> np.ndarray:

        """
        Load all images belonging to one ID

        Parameters:
            idx (int | str): Cluster ID or page ID.

        Returns:
            numpy.ndarray: Stacked images.
        """

        if self.page:
            images = []

            for line_id in self.id_lines[idx]:
                image = self._read_line(line_id)

                # in page mode one line can produce multiple windows, so extend the list with all returned windows.
                if image is not None:
                    images += image

            if len(images) == 0:
                raise ValueError(f"No images for page id: {idx}")

            images = np.stack(images, axis=0)

        else:
            images = [self._read_line(line_name) for line_name in self.id_lines[idx]]
            images = np.stack(images, axis=0)

        return images

    @staticmethod
    def convert_line_id_to_page_id(line_id: str) -> str:

        """
        Convert a line ID into a page ID.

        Parameters:
            line_id (str): Line identifier.

        Returns:
            str: Page identifier.
        """

        not_uuid_pattern = re.compile(r"r\d+-l\d+\.jpg$")

        if not_uuid_pattern.search(line_id):
            # for non-UUID naming, remove the trailing row/line suffix.
            page_id = "-".join(line_id.split("-")[:-2])

        else:
            # for UUID naming, the first UUID is treated as page ID.
            line_id_without_extension = line_id[:-4]
            splits = line_id_without_extension.split("-")

            uuid_1 = "-".join(splits[:5])
            uuid_2 = "-".join(splits[5:])

            try:
                uuid.UUID(uuid_1)
            except ValueError:
                logger.warning(f"Invalid UUID: {uuid_1} in line_id: {line_id}")

            try:
                uuid.UUID(uuid_2)
            except ValueError:
                logger.warning(f"Invalid UUID: {uuid_2} in line_id: {line_id}")

            page_id = uuid_1

        return page_id

    def __getitem__(self, idx: int):

        """
        Return a positive pair of images from the same ID.

        Parameters:
            idx (int): Dataset index.

        Returns:
            tuple: (image_1, image_2, line_cluster_id)

        Raises:
            ValueError: when no transform is provided for the patches of each image
        """

        image_1 = self._read_line(self.lines[idx][1])
        line_cluster_id = self.lines[idx][0]

        # pick another sample from the same class to create a positive pair.
        line_name = np.random.choice(self.id_lines[line_cluster_id])
        image_2 = self._read_line(line_name)

        if self.page:
            # in page mode _read_line returns a list of windows
            # original behavior uses only the first window
            image_1 = image_1[0]
            image_2 = image_2[0]

        if self.aug is not None:
            image_1, image_2 = self.aug(images=[image_1, image_2])

        # patch both images
        image_1 = self.patcher.extract_patches(image_1)
        image_2 = self.patcher.extract_patches(image_2)

        if self.transform is None:
            raise ValueError("transform must be provided when using patched images.")

        # convert each patch separately: (N, H, W, C) -> (N, C, H, W)
        image_1 = torch.stack([self.transform(patch) for patch in image_1], dim=0)
        image_2 = torch.stack([self.transform(patch) for patch in image_2], dim=0)

        return image_1, image_2, line_cluster_id

    def get_characters(self) -> list:

        """
        Return unique characters appearing in stored image names.

        Returns:
            list: Unique characters.
        """

        return list(set("".join([item[1] for item in self.lines])))
