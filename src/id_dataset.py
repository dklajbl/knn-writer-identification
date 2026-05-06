import lmdb
import random
import logging
import math
from collections import defaultdict

import cv2
import numpy as np
np.bool = bool

import imgaug.augmenters as iaa
import torch
import torch.utils.data

from src.patchers.patcher_config import PatcherConfig
from src.patchers.make_patcher import make_patcher


logger = logging.getLogger(__name__)


class AuthorStratifiedBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler that guarantees exactly ``num_authors`` distinct authors appear
    in every batch, each represented by exactly ``batch_size // num_authors`` samples.
    Every sample is used at most once per epoch.

    The sampler works on the flat ``dataset.lines`` list that is rebuilt by
    ``IdDataset.resample_epoch()``.  Call ``resample_epoch()`` on the dataset
    **before** constructing a new DataLoader (or re-iterating this sampler)
    each epoch.

    Parameters:
        dataset (IdDataset): The dataset whose ``.lines`` and ``.id_lines``
            attributes describe the active epoch samples.
        batch_size (int): Total number of samples per batch.
        num_authors (int): Exact number of distinct authors per batch.
            Must evenly divide ``batch_size``.
        drop_last (bool): If True, the last incomplete batch is dropped.
    """

    def __init__(
        self,
        dataset: "IdDataset",
        batch_size: int,
        num_authors: int,
        drop_last: bool = True,
    ):
        super().__init__()

        if batch_size % num_authors != 0:
            raise ValueError(
                f"num_authors ({num_authors}) must evenly divide batch_size ({batch_size})."
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_authors = num_authors
        self.samples_per_author = batch_size // num_authors
        self.drop_last = drop_last

    def _build_slots(self) -> list[tuple[int, list[int]]]:
        """
        Build a shuffled list of slots where each slot is
        (author_id, [samples_per_author indices]).
        Authors with fewer samples than samples_per_author are excluded.
        Leftover samples that don't fill a complete slot are dropped.
        """
        author_to_indices: dict[int, list[int]] = defaultdict(list)
        for flat_idx, (cluster_id, _) in enumerate(self.dataset.lines):
            author_to_indices[cluster_id].append(flat_idx)

        slots: list[tuple[int, list[int]]] = []
        for author, idxs in author_to_indices.items():
            if len(idxs) < self.samples_per_author:
                continue
            random.shuffle(idxs)
            for i in range(0, len(idxs) - len(idxs) % self.samples_per_author, self.samples_per_author):
                slots.append((author, idxs[i : i + self.samples_per_author]))

        random.shuffle(slots)
        return slots

    def __len__(self) -> int:
        n = len(self.dataset.lines)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self):
        slots = self._build_slots()

        while True:
            # collect one slot per author until we have num_authors distinct authors
            batch_slots: list[tuple[int, list[int]]] = []
            used_authors: set[int] = set()
            skipped: list[tuple[int, list[int]]] = []

            for slot in slots:
                if len(batch_slots) == self.num_authors:
                    skipped.append(slot)
                elif slot[0] not in used_authors:
                    batch_slots.append(slot)
                    used_authors.add(slot[0])
                else:
                    skipped.append(slot)

            if len(batch_slots) < self.num_authors:
                # not enough distinct authors left to form a full batch
                break

            slots = skipped

            batch = [idx for _, idxs in batch_slots for idx in idxs]
            random.shuffle(batch)
            yield batch


class IdDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset class for loading line images from an LMDB database.

    Each sample is a single line image identified by (cluster_id, image_name).
    Images are loaded from LMDB, optionally augmented, patched, and converted to tensors.

    Per-epoch resampling
    --------------------
    When ``samples_per_author`` is set, call ``resample_epoch()`` at the start
    of each training epoch to rebuild ``self.lines`` with a fresh selection of
    samples for every author.  Authors with fewer images than
    ``samples_per_author`` contribute all their images (no repetition).
    A usage counter tracks how many times each (author, image) pair has been
    selected; images that have been used less often are preferred on the next
    draw.
    """

    def __init__(
        self,
        file_name: str,
        lmdb_path: str,
        augment: bool = True,
        restrict_data: bool = False,
        test: bool = False,
        patcher_config: PatcherConfig | None = None,
        samples_per_author: int | None = None,
        data_fraction: float = 1.0,
    ):
        """
        Initialize the dataset.

        Parameters:
            file_name (str): Path to the file containing image names and IDs.
            lmdb_path (str): Path to the LMDB database.
            augment (bool): Whether to use image augmentation.
            restrict_data (bool): Whether to restrict dataset size.
            test (bool): Whether the dataset is used in test mode.
            patcher_config (PatcherConfig): Patcher configuration.
            samples_per_author (int | None): How many samples to draw per author
                each epoch.  None disables per-epoch resampling (original
                behaviour).
            data_fraction (float): Fraction of dataset to use for training

        Raises:
            ValueError: if patcher_config is not provided.
        """

        super().__init__()

        if patcher_config is None:
            raise ValueError("patcher_config must be provided. Images are always patched.")

        self.file_name = file_name
        self.lmdb_path = lmdb_path
        self.augment = augment
        self.restrict_data = restrict_data
        self.test = test
        self.samples_per_author = samples_per_author
        self.data_fraction = data_fraction

        self.patcher = make_patcher(patcher_config)

        # LMDB objects are initialised lazily to avoid multiprocessing issues.
        self.env = None
        self.txn = None

        self.aug = None  # augmentation pipeline

        # list of (cluster_id, image_name) — the active epoch sample list
        self.lines: list[tuple[int, str]] = []

        # cluster_id -> list[image_name] — full catalogue (never modified after load)
        self.id_lines: dict[int, list[str]] = defaultdict(list)

        # usage counters: cluster_id -> {image_name: int}
        # incremented each time an image is included in a resampled epoch
        self._usage_counts: dict[int, dict[str, int]] = {}

        self._load_lines()

        if self.data_fraction < 1.0:
            self._subsample_authors(self.data_fraction)

        if self.augment:
            self.uniformize_data_distribution()

        if self.restrict_data:
            self.restrict_data_distribution()

        # initialise usage counters to 0 for every known image
        for cluster_id, names in self.id_lines.items():
            self._usage_counts[cluster_id] = {name: 0 for name in names}

    # ------------------------------------------------------------------
    # loading & distribution helpers (unchanged from original)
    # ------------------------------------------------------------------

    def _load_lines(self) -> None:
        """
        Load lines from the input file and group them by ID.

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

        raw_lines = [
            (int(parts[1]), parts[0])
            for parts in parsed_lines
            if len(parts) == 2
        ]

        # remap cluster_ids to contiguous 0-based indices
        unique_ids = sorted(set(cluster_id for cluster_id, _ in raw_lines))
        self._id_remap = {original: remapped for remapped, original in enumerate(unique_ids)}

        self.lines = [
            (self._id_remap[cluster_id], image_name)
            for cluster_id, image_name in raw_lines
        ]

        for cluster_id, image_name in self.lines:
            self.id_lines[cluster_id].append(image_name)

    def _subsample_authors(self, fraction: float) -> None:

        """
        Randomly retain a fraction of authors from the loaded catalogue.

        Parameters:
            fraction (float): Proportion of authors to keep (0.0–1.0). At least 2 authors are always kept.
        """

        all_ids = list(self.id_lines.keys())
        keep_n = max(2, int(len(all_ids) * fraction))
        kept_ids = set(random.sample(all_ids, keep_n))

        self.id_lines = {cid: names for cid, names in self.id_lines.items() if cid in kept_ids}
        self.lines = [(cid, name) for cid, name in self.lines if cid in kept_ids]

        # re-remap to contiguous 0-based indices to keep labels valid for ArcFaceLoss
        remap = {old: new for new, old in enumerate(sorted(kept_ids))}
        self.id_lines = {remap[cid]: names for cid, names in self.id_lines.items()}
        self.lines = [(remap[cid], name) for cid, name in self.lines]

    def _ensure_lmdb_open(self) -> None:
        """
        Open LMDB database if it is not already open.
        LMDB is opened lazily because DataLoader may spawn multiple worker processes.
        """

        if self.txn is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
            )
            self.txn = self.env.begin(write=False)

        if self.augment and self.aug is None:
            self.aug = self._build_augmenter()

    @staticmethod
    def _build_augmenter() -> iaa.SomeOf:
        """
        Create the imgaug augmentation pipeline.

        Returns:
            iaa.SomeOf: Random augmentation pipeline.
        """

        return iaa.SomeOf(
            n=(1, 4),
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

        logger.info(
            f"{self.file_name} -> restrict data distribution from "
            f"{len(self.id_lines)} ids, {len(self.lines)} lines"
        )

        new_id_lines = defaultdict(list)

        for cluster_id in self.id_lines.keys():
            if len(self.id_lines[cluster_id]) >= min_id_size:
                shuffled_names = self.id_lines[cluster_id][:]
                random.shuffle(shuffled_names)
                new_id_lines[cluster_id] = shuffled_names[:max_id_size]

        items = list(new_id_lines.items())
        random.shuffle(items)

        if max_id != -1:
            items = items[:max_id]

        self.id_lines = dict(items)

        self.lines = []
        for cluster_id in self.id_lines:
            self.lines += [(cluster_id, name) for name in self.id_lines[cluster_id]]

        logger.info(
            f"{self.file_name} -> max_id:{max_id}, max_id_size:{max_id_size}, "
            f"min_id_size:{min_id_size}, restricted to {len(self.id_lines)} ids, "
            f"{len(self.lines)} lines"
        )

    def uniformize_data_distribution(self, p: float = 0.65) -> None:
        """
        Oversample smaller classes to reduce class imbalance.

        Parameters:
            p (float): Rebalancing strength.
        """

        ids = sorted([key for key in self.id_lines])
        counts = np.asarray([len(self.id_lines[key]) for key in ids])
        max_count = np.max(counts)

        new_counts = (counts ** p / max_count ** p) * max_count

        logger.info(f"{self.file_name} -> uniformize data distribution from {len(self.lines)} lines")

        for cluster_id, old_count, new_count in zip(ids, counts, new_counts):
            new_count = int(new_count)
            for _ in range(new_count - old_count):
                self.lines.append((cluster_id, random.choice(self.id_lines[cluster_id])))

        logger.info(
            f"{self.file_name} -> p: {p}, max_count:{max_count}, "
            f"uniformize data distribution to {len(self.lines)} lines"
        )

    # ------------------------------------------------------------------
    # per-epoch resampling
    # ------------------------------------------------------------------

    def resample_epoch(self) -> None:
        """
        Rebuild ``self.lines`` for the next training epoch.

        Behaviour
        ---------
        * For each author the method draws ``samples_per_author`` images without
          replacement, using **inverse-usage weights**: images that have been
          included in fewer previous epochs are proportionally more likely to be
          selected.
        * Authors whose catalogue is smaller than ``samples_per_author`` contribute
          all their images exactly once — no repetition, no padding.
        * The resulting flat sample list is **shuffled** so authors are spread
          randomly across the epoch rather than grouped together.
        * Usage counters are updated after selection.

        If ``samples_per_author`` is None this method is a no-op, preserving the
        original dataset behaviour.
        """

        if self.samples_per_author is None:
            return

        new_lines: list[tuple[int, str]] = []

        for cluster_id, all_names in self.id_lines.items():
            n_available = len(all_names)
            n_wanted = self.samples_per_author

            if n_available <= n_wanted:
                # take every available image once — no repetition
                chosen = list(all_names)
            else:
                # build sampling weights: weight = 1 / (usage_count + 1)
                # so never-used images have weight 1, once-used have 0.5, etc.
                counts = self._usage_counts[cluster_id]
                weights = np.array(
                    [1.0 / (counts.get(name, 0) + 1) for name in all_names],
                    dtype=np.float64,
                )
                weights /= weights.sum()  # normalise to a probability distribution

                chosen_indices = np.random.choice(
                    n_available,
                    size=n_wanted,
                    replace=False,
                    p=weights,
                )
                chosen = [all_names[i] for i in chosen_indices]

            # update usage counters
            for name in chosen:
                self._usage_counts[cluster_id][name] = (
                    self._usage_counts[cluster_id].get(name, 0) + 1
                )

            new_lines.extend((cluster_id, name) for name in chosen)

        # shuffle so samples from different authors are interleaved
        random.shuffle(new_lines)
        self.lines = new_lines

        logger.info(
            f"{self.file_name} -> resampled epoch: "
            f"{len(self.id_lines)} authors, {len(self.lines)} total samples "
            f"(samples_per_author={self.samples_per_author})"
        )

    # ------------------------------------------------------------------
    # standard dataset interface
    # ------------------------------------------------------------------

    def id_count(self) -> int:
        """Return the number of distinct IDs."""
        return np.max(list(self.id_lines)) + 1

    def __len__(self) -> int:
        return len(self.lines)

    def _read_line(self, name: str) -> np.ndarray | None:
        """
        Load image from LMDB and convert it to a numpy array.

        Parameters:
            name (str): Key of the image inside LMDB.

        Returns:
            np.ndarray | None: Image of shape (H, W, 1) or None on failure.
        """

        self._ensure_lmdb_open()

        data = self.txn.get(name.encode())

        if data is None:
            logger.warning(
                f"Unable to load image '{name}' specified in '{self.file_name}' "
                f"from DB '{self.lmdb_path}'."
            )
            return None

        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        if image is None:
            logger.warning(f"Unable to decode image '{name}'.")
            return None

        return np.expand_dims(image, axis=-1)  # (H, W, 1)

    def get_single_id_lines(self, idx: int, line_count: int = 32) -> list[np.ndarray] | None:
        """
        Load a random subset of images for one ID.

        Parameters:
            idx (int): Cluster ID.
            line_count (int): Maximum number of images to load.

        Returns:
            list[np.ndarray] | None: List of images or None.
        """

        if idx not in self.id_lines:
            return None

        lines = np.random.choice(
            self.id_lines[idx],
            size=min(line_count, len(self.id_lines[idx])),
            replace=False,
        )

        images = [self._read_line(line_name) for line_name in lines]
        images = [img for img in images if img is not None]

        return images if images else None

    def get_single_id_all_lines(self, idx: int) -> list[np.ndarray]:
        """
        Load all images belonging to one ID.

        Parameters:
            idx (int): Cluster ID.

        Returns:
            list[np.ndarray]: List of images.
        """

        images = [self._read_line(name) for name in self.id_lines[idx]]
        images = [img for img in images if img is not None]

        if not images:
            raise ValueError(f"No images could be loaded for id: {idx}")

        return images

    def get_original_image1(self, idx: int) -> torch.Tensor:
        """
        Return an original image without patching (augmentation still applied).

        Parameters:
            idx (int): Dataset index.

        Returns:
            torch.Tensor: Image tensor of shape (C, H, W).
        """

        image = self._read_line(self.lines[idx][1])

        if self.aug is not None:
            image = self.aug(images=[image])[0]

        return torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Return a positive pair of images from the same author.

        Parameters:
            idx (int): Dataset index.

        Returns:
            tuple[torch.Tensor, torch.Tensor, int]: (image_1, image_2, cluster_id)
        """

        cluster_id, name_1 = self.lines[idx]
        image_1 = self._read_line(name_1)

        if image_1 is None:
            raise RuntimeError(
                f"Failed to load image '{name_1}' (index {idx}) from LMDB '{self.lmdb_path}'."
            )

        # pick another sample from the same author for the positive pair
        name_2 = np.random.choice(self.id_lines[cluster_id])
        image_2 = self._read_line(name_2)

        if image_2 is None:
            raise RuntimeError(
                f"Failed to load image '{name_2}' (positive pair for index {idx}) "
                f"from LMDB '{self.lmdb_path}'."
            )

        if self.aug is not None:
            image_1, image_2 = self.aug(images=[image_1, image_2])

        image_1 = self.patcher.extract_patches(image_1, key=name_1)
        image_2 = self.patcher.extract_patches(image_2, key=name_2)

        image_1 = torch.from_numpy(image_1.transpose(0, 3, 1, 2).copy()).float().div_(255.0)
        image_2 = torch.from_numpy(image_2.transpose(0, 3, 1, 2).copy()).float().div_(255.0)

        return image_1, image_2, cluster_id

    def get_characters(self) -> list:
        """Return unique characters appearing in stored image names."""
        return list(set("".join([item[1] for item in self.lines])))
