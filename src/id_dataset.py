import logging
import re
import sys
import uuid
from collections import defaultdict

import cv2
import lmdb
import numpy as np
import torch
import random

import imgaug.augmenters as iaa

logger = logging.getLogger(__name__)


class IdDataset(torch.utils.data.Dataset):
    def __init__(self,
                 file_name,
                 lmdb_path,
                 width=320,
                 transform=None,
                 augment=True,
                 restrict_data=False,
                 test=False,
                 page=False):
        super().__init__()
        self.width = width
        self.file_name = file_name
        self.lmdb_path = lmdb_path
        self.txn = None
        self.transform = transform

        self.augment = augment
        self.restrict_data = restrict_data
        self.aug = None

        self.test = test
        self.page = page

        with open(self.file_name, 'r', encoding="utf-8") as f:
            lines = f.readlines()

        if not self.page:
            lines = [line.split()[:2] for line in lines]
            self.lines = [(int(line[1]), line[0])
                          for line in lines if len(line) == 2]
            self.id_lines = defaultdict(list)
            for i, name in self.lines:
                self.id_lines[i].append(name)
        else:
            line_ids = [line.split()[0].strip() for line in lines]
            self.lines = [(self.convert_line_id_to_page_id(
                line_id), line_id) for line_id in line_ids]
            self.id_lines = defaultdict(list)
            for page_id, line_id in self.lines:
                self.id_lines[page_id].append(line_id)

        if self.augment:
            self.uniformer_data_distribution()
        if self.restrict_data:
            self.restrict_data_distribution()

    def restrict_data_distribution(self, max_id=-1, max_id_size=5000, min_id_size=0):
        logger.info(
            f"{self.file_name} -> restrict data distribution from {len(self.id_lines)} ids, {len(self.lines)} lines")
        new_id_lines = defaultdict(list)
        for i in self.id_lines.keys():
            if len(self.id_lines[i]) >= min_id_size:
                random.shuffle(self.id_lines[i])
                new_id_lines[i] = self.id_lines[i][:max_id_size]
        new_id_lines = list(new_id_lines.items())
        random.shuffle(new_id_lines)
        self.id_lines = dict(new_id_lines[:max_id])
        self.lines = []
        for i in self.id_lines:
            self.lines += [(i, name) for name in self.id_lines[i]]
        logger.info(f"{self.file_name} -> max_id:{max_id}, max_id_size:{max_id_size}, min_id_size:{min_id_size}, "
                    f"restricted data distribution to {len(self.id_lines)} ids, {len(self.lines)} lines")

    def uniformer_data_distribution(self, p=0.65):
        ids = sorted([k for k in self.id_lines])
        counts = np.asarray([len(self.id_lines[k]) for k in ids])
        max_count = np.max(counts)
        new_counts = (counts ** p / max_count ** p) * max_count
        logger.info(
            f"{self.file_name} -> uniformer data distribution from {len(self.lines)} lines")
        for i, c_old, c_new in zip(ids, counts, new_counts):
            c_new = int(c_new)
            logger.debug(i, c_old, c_new, c_new - c_old)
            for j in range(c_new - c_old):
                self.lines.append((i, np.random.choice(self.id_lines[i])))
        logger.info(
            f"{self.file_name} -> p: {p}, max_count:{max_count}, "
            f"uniformer data distribution to {len(self.lines)} lines")

    def id_count(self):
        if self.page:
            return len(self.id_lines.keys())
        return np.max(list(self.id_lines)) + 1

    def __len__(self):
        return len(self.lines)

    def _read_line(self, name):
        data = self.txn.get(name.encode())
        if data is None:
            logger.warning(
                f"Unable to load image '{name}' specified in '{self.file_name}' from DB '{self.lmdb_path}'.",
                file=sys.stderr)
            return None
        image = cv2.imdecode(np.frombuffer(
            data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning(
                f"Unable to decode image '{name}'.", file=sys.stderr)
            return None
        if not self.page:
            if image.shape[1] > self.width:
                if self.test:
                    pos = image.shape[1] // 2 - self.width // 2
                else:
                    pos = np.random.randint(0, image.shape[1] - self.width)
                image = image[:, pos:][:, :self.width]
            elif image.shape[1] < self.width:
                data = np.zeros(
                    [image.shape[0], self.width, 3], dtype=np.uint8)
                data[:, :image.shape[1]] = image
                image = data
            return image
        else:
            if image.shape[1] > self.width:
                images = []
                pos = 0
                while True:
                    new_image = image[:, pos:pos + self.width]
                    images.append(new_image)
                    if pos == image.shape[1] - self.width:
                        break
                    pos = min(pos + 50, image.shape[1] - self.width)
            elif image.shape[1] < self.width:
                data = np.zeros(
                    [image.shape[0], self.width, 3], dtype=np.uint8)
                data[:, :image.shape[1]] = image
                images = [data]
            else:
                images = [image]
            return images

    def get_single_id_lines(self, idx, line_count=32):
        if idx not in self.id_lines:
            return None
        lines = np.random.choice(self.id_lines[idx], size=min(
            line_count, len(self.id_lines[idx])), replace=False)
        images = [self._read_line(line_name) for line_name in lines]
        images = np.stack(images, axis=0)
        return images

    def get_single_id_all_lines(self, idx):
        if self.page:
            images = []
            for line_id in self.id_lines[idx]:
                image = self._read_line(line_id)
                if image is not None:
                    images += image
            # logger.info(f"Page id: {idx}, line count: {len(images)}")
            if len(images) == 0:
                raise ValueError(f"No images for page id: {idx}")
            images = np.stack(images, axis=0)
        else:
            images = [self._read_line(line_name)
                      for line_name in self.id_lines[idx]]
            images = np.stack(images, axis=0)
        return images

    @staticmethod
    def convert_line_id_to_page_id(line_id):
        not_uuid_pattern = re.compile(r"r\d+-l\d+\.jpg$")
        if not_uuid_pattern.search(line_id):
            page_id = '-'.join(line_id.split('-')[:-2])
        else:
            line_id = line_id[:-4]
            splits = line_id.split('-')
            uuid1 = '-'.join(splits[:5])
            uuid2 = '-'.join(splits[5:])
            try:
                uuid.UUID(uuid1)
            except ValueError:
                logger.warning(f"Invalid UUID: {uuid1} in line_id: {line_id}")
            try:
                uuid.UUID(uuid2)
            except ValueError:
                logger.warning(f"Invalid UUID: {uuid2} in line_id: {line_id}")
            page_id = uuid1
        return page_id

    def __getitem__(self, idx):
        if self.txn is None:
            env = lmdb.open(self.lmdb_path)
            self.txn = env.begin()
            if self.augment:
                self.aug = iaa.SomeOf(n=(1, 4), children=[
                    iaa.convolutional.DirectedEdgeDetect(
                        alpha=(0.05, 0.2), direction=(0.0, 1.0)),
                    iaa.convolutional.EdgeDetect(alpha=(0.05, 0.15)),
                    iaa.convolutional.Emboss(
                        alpha=(0.05, 0.2), strength=(0.2, 0.7)),
                    iaa.convolutional.Sharpen(
                        alpha=(0.05, 0.2), lightness=(0.8, 1.2)),

                    iaa.color.AddToHue(value=(-64, 64)),
                    iaa.color.AddToBrightness(add=(-40, 40)),
                    iaa.color.AddToSaturation(value=(-64, 64)),
                    iaa.color.Grayscale(),
                    iaa.color.Grayscale(),
                    iaa.color.MultiplyBrightness(mul=(0.8, 1.2)),
                    iaa.color.MultiplyHue(mul=(-0.7, 0.7)),
                    iaa.color.MultiplySaturation(mul=(0.0, 2.0)),
                    iaa.color.Posterize(nb_bits=(2, 8)),

                    iaa.contrast.AllChannelsCLAHE(clip_limit=(
                        0.1, 8), tile_grid_size_px=(3, 12), tile_grid_size_px_min=3),

                    iaa.contrast.CLAHE(clip_limit=(0.1, 8), tile_grid_size_px=(
                        3, 12), tile_grid_size_px_min=3),
                    iaa.contrast.GammaContrast(gamma=(0.6, 1.8)),
                    iaa.contrast.LogContrast(gain=(0.6, 1.4)),

                    iaa.BlendAlpha(
                        (0.2, 0.7), iaa.contrast.AllChannelsHistogramEqualization()),
                    iaa.BlendAlpha(
                        (0.2, 0.7), iaa.contrast.HistogramEqualization()),

                    iaa.blur.BilateralBlur(d=(1, 7), sigma_color=(
                        10, 250), sigma_space=(10, 250)),
                    iaa.blur.GaussianBlur(sigma=(0.0, 2.5)),

                    iaa.pillike.Solarize(p=1.0, threshold=128),
                    iaa.pillike.EnhanceColor(factor=(0.5, 1.5)),
                    iaa.pillike.EnhanceContrast(factor=(0.5, 1.5)),
                    iaa.pillike.EnhanceBrightness(factor=(0.5, 1.5)),
                    iaa.pillike.EnhanceSharpness(factor=(0.5, 1.5)),
                    iaa.pillike.FilterEdgeEnhance(),
                    iaa.pillike.FilterSharpen(),
                    iaa.pillike.FilterDetail()
                ])

        image1 = self._read_line(self.lines[idx][1])
        line_cluster_id = self.lines[idx][0]
        line_name = np.random.choice(self.id_lines[line_cluster_id])
        image2 = self._read_line(line_name)
        if self.page:
            image1 = image1[0]
            image2 = image2[0]

        if self.aug is not None:
            image1, image2 = self.aug(images=[image1, image2])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, line_cluster_id

    def get_characters(self):
        return list(set(''.join([x[1] for x in self.lines])))
