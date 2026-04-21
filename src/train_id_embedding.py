import argparse
import logging
import os
import sys
import time
from contextlib import nullcontext

import numpy as np


np.bool = bool

import torch
from torch.utils.data import DataLoader

from pytorch_metric_learning import losses

from src.model import WriterIdentificationEncoder
from src.id_dataset import IdDataset, AuthorStratifiedBatchSampler
from src.utils import convert_sec_to_hours_minutes_seconds

from src.patchers.patcher_config import PatcherConfig, PATCH_METHODS
from src.patchers.collate import pad_patches_collate
from src.eval import test_identification

from src.env_vars import NP_RANDOM_SEED


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser(usage="Trains contrastive self-supervised training on artificial data.")

    parser.add_argument(
        "--gt-file",
        required=True,
        help="Text file with an image file name and id on each line."
    )
    parser.add_argument(
        "--gt-file-gallery",
        help="Gallery text file with an image file name and id on each line."
    )
    parser.add_argument(
        "--gt-file-query",
        help="Query text file with an image file name and id on each line."
    )
    parser.add_argument(
        "--lmdb",
        required=True,
        help="Path to LMDB database."
    )

    parser.add_argument(
        "--patcher",
        type=str,
        choices=PATCH_METHODS,
        default="grid",
        help=f"Patching method to use. Available options: {PATCH_METHODS}",
    )

    parser.add_argument("--patch-count", type=int, default=50)

    parser.add_argument(
        "--patch-height",
        type=int,
        default=32,
        help="Patch height in pixels used by all patchers."
    )

    parser.add_argument(
        "--patch-width",
        type=int,
        default=32,
        help="Patch width in pixels used by all patchers."
    )

    parser.add_argument("--start-iteration", default=0, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument(
        "--view-step",
        default=50,
        type=int,
        help="Number of training iterations between evaluations."
    )

    parser.add_argument(
        "--embed-dim",
        default=256,
        type=int,
        help="Output embedding dimension."
    )
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=0.0002, type=float)
    parser.add_argument("--weight-decay", default=0.01, type=float)

    parser.add_argument(
        "--temperature",
        default=0.5,
        type=float,
        help="Temperature for NTXent loss."
    )
    parser.add_argument("--out-checkpoints-dir", default='.', type=str)
    parser.add_argument("--show-dir", default='.', type=str)

    parser.add_argument("--num-workers", default=4, type=int, help="Number of DataLoader worker processes.")
    parser.add_argument("--eval-on-start", action="store_true")
    parser.add_argument("--logging-level", default="INFO")

    parser.add_argument(
        "--samples-per-author",
        type=int,
        default=None,
        help=(
            "Number of samples drawn per author per epoch. "
            "Authors with fewer images contribute all their images (no repetition). "
            "If omitted, the original dataset behaviour is preserved."
        ),
    )
    parser.add_argument(
        "--min-authors-per-batch",
        type=int,
        default=2,
        help=(
            "Minimum number of distinct authors that must appear in every "
            "training batch. Must be <= batch-size. Default: 2."
        ),
    )

    return parser.parse_args()

def log_args(args: argparse.Namespace, logger=None) -> None:
    """Print parsed arguments, one per line, aligned by the longest name."""
    items = vars(args).items()
    width = max(len(k) for k, _ in items)
    lines = ["Run arguments:"]
    for key, val in items:
        lines.append(f"  {key.replace('_', '-'):<{width}}  {val}")
    msg = "\n".join(lines)
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

def configure_logging(logging_level: str) -> None:
    """
    Configure root logger.

    Parameters:
        logging_level (str): Logging level as string.
    """

    log_formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")
    log_formatter.converter = time.gmtime

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(log_formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(handler)
    root_logger.setLevel(logging_level)


def create_model(args, device: torch.device) -> torch.nn.Module:
    """
    Create encoder model and optionally load a checkpoint.

    Parameters:
        args (argparse.Namespace): Parsed arguments.
        device (torch.device): Device used for computation.

    Returns:
        torch.nn.Module: Initialised encoder model.
    """

    image_encoder = WriterIdentificationEncoder(
        in_channels=1,
        hidden_dim=256,
        embed_dim=args.embed_dim,
        nhead=8,
        num_transformer_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
        use_positional_encoding=(args.patcher == "grid"),
    ).to(device)

    if args.start_iteration > 0:
        checkpoint_path = os.path.join(args.out_checkpoints_dir, f"cp-{args.start_iteration:07d}.img.ckpt")
        logger.info(f"Loading image checkpoint {checkpoint_path}")
        image_encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))

    return image_encoder


def create_train_dataset(args) -> IdDataset:
    """
    Create the training dataset.

    The dataset is created once and reused across epochs.  Call
    ``dataset.resample_epoch()`` before each epoch to get a fresh sample
    selection when ``--samples-per-author`` is set.

    Parameters:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        IdDataset: Training dataset.
    """

    patcher_config = PatcherConfig(
        method=args.patcher,
        patch_count=args.patch_count,
        random_seed=NP_RANDOM_SEED,
        patch_height=args.patch_height,
        patch_width=args.patch_width,
        min_partial_ratio=0.3,
    )

    return IdDataset(
        args.gt_file,
        args.lmdb,
        augment=True,
        patcher_config=patcher_config,
        samples_per_author=args.samples_per_author,
        min_authors_per_batch=args.min_authors_per_batch,
    )


def create_train_dataloader(args, train_dataset: IdDataset) -> DataLoader:
    """
    Build a DataLoader for one epoch using an ``AuthorStratifiedBatchSampler``.

    This must be called **after** ``train_dataset.resample_epoch()`` so the
    sampler sees the up-to-date ``dataset.lines``.

    Parameters:
        args (argparse.Namespace): Parsed arguments.
        train_dataset (IdDataset): Dataset whose ``.lines`` reflect the current epoch.

    Returns:
        DataLoader: Ready-to-iterate training dataloader.
    """

    collate_fn = pad_patches_collate if args.patcher == "grid" else None

    batch_sampler = AuthorStratifiedBatchSampler(
        dataset=train_dataset,
        batch_size=args.batch_size,
        min_authors=args.min_authors_per_batch,
        drop_last=True,
    )

    return DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )


def create_eval_dataloaders(args) -> tuple[DataLoader | None, DataLoader | None]:
    """
    Create gallery and query dataloaders used for evaluation.

    Parameters:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        tuple[DataLoader | None, DataLoader | None]: Gallery and query dataloaders.
    """

    if not (args.gt_file_gallery and args.gt_file_query):
        return None, None

    patcher_config = PatcherConfig(
        method=args.patcher,
        patch_count=args.patch_count,
        random_seed=NP_RANDOM_SEED,
        patch_height=args.patch_height,
        patch_width=args.patch_width,
        min_partial_ratio=0.3,
    )

    collate_fn = pad_patches_collate if args.patcher == "grid" else None

    gallery_dataset = IdDataset(
        args.gt_file_gallery,
        args.lmdb,
        augment=False,
        restrict_data=False,
        test=True,
        patcher_config=patcher_config,
    )
    gallery_dataloader = DataLoader(
        gallery_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    query_dataset = IdDataset(
        args.gt_file_query,
        args.lmdb,
        augment=False,
        restrict_data=False,
        test=True,
        patcher_config=patcher_config,
    )
    query_dataloader = DataLoader(
        query_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return gallery_dataloader, query_dataloader


def train_one_step(
    image_encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_optimizer: torch.optim.Optimizer,
    loss_object,
    images_1: torch.Tensor,
    images_2: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    mask_1: torch.Tensor | None = None,
    mask_2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """
    Perform one training step.

    Expected input shapes:
        images_1: [batch_size, patch_count, channels, height, width]
        images_2: [batch_size, patch_count, channels, height, width]
        labels:   [batch_size]
        mask_1:   [batch_size, patch_count] (optional, bool, True = padding)
        mask_2:   [batch_size, patch_count] (optional, bool, True = padding)

    Parameters:
        image_encoder (torch.nn.Module): Encoder model.
        optimizer (torch.optim.Optimizer): Optimizer.
        loss_optimizer (torch.optim.Optimizer):
        loss_object: Metric learning loss object.
        images_1 (torch.Tensor): First batch of patched images.
        images_2 (torch.Tensor): Second batch of patched images.
        labels (torch.Tensor): Batch labels.
        device (torch.device): Computation device.
        scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed precision.
        mask_1 (torch.Tensor, optional): Padding mask for images_1.
        mask_2 (torch.Tensor, optional): Padding mask for images_2.

    Returns:
        tuple[torch.Tensor, float]: Embeddings and scalar loss value.
    """

    # move images and labels to target device
    images_1 = images_1.to(device)
    images_2 = images_2.to(device)
    labels = labels.to(device)

    images = torch.cat([images_1, images_2], dim=0)
    labels = torch.cat([labels, labels], dim=0)

    # concatenate padding masks the same way (None when all patches are real)
    padding_mask = None
    if mask_1 is not None and mask_2 is not None:
        padding_mask = torch.cat([mask_1, mask_2], dim=0).to(device)

    optimizer.zero_grad(set_to_none=True)
    loss_optimizer.zero_grad(set_to_none=True)

    use_amp = device.type == "cuda"

    # encoder forward pass in float16
    if use_amp and scaler is not None:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            embedding = image_encoder(images, padding_mask=padding_mask)

        # loss in float32 — ArcFace is not compatible with float16
        loss = loss_object(embedding.float(), labels)

        # backward through scaler (covers encoder gradients)
        scaler.scale(loss).backward()

        # encoder step via scaler
        scaler.step(optimizer)
        scaler.update()

        # proxy weights step without scaler — they live in float32 already
        loss_optimizer.step()
    else:
        embedding = image_encoder(images, padding_mask=padding_mask)
        loss = loss_object(embedding, labels)
        loss.backward()
        optimizer.step()
        loss_optimizer.step()

    return embedding, loss.item()


def main() -> None:
    """
    Main training entry point.
    """
    time_start = time.time()
    args = parse_args()

    configure_logging(args.logging_level)

    # logger.info("\t\t\n".join(sys.argv))
    # logger.info(f"ARGS {args}")
    log_args(args, logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # spawn start method is often safer when DataLoader uses workers.
    torch.multiprocessing.set_start_method("spawn")

    image_encoder = create_model(args, device)

    # create the training dataset once; it is resampled at the start of each epoch
    train_dataset = create_train_dataset(args)

    gallery_dataloader, query_dataloader = create_eval_dataloaders(args)

    loss_object = losses.ArcFaceLoss(
        num_classes=len(train_dataset.id_lines),
        embedding_size=args.embed_dim,
        margin=28.6,
        scale=64,
    ).to(device)

    loss_optimizer = torch.optim.AdamW(
        loss_object.parameters(),
        lr=args.learning_rate * 0.1,
        weight_decay=args.weight_decay,
    )

    optimizer = torch.optim.AdamW(
        image_encoder.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    epoch = 0

    time_setup_finished = time.time()

    while True:
        epoch += 1
        epoch_start = time.time()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        # --- resample dataset and rebuild dataloader for this epoch ---
        train_dataset.resample_epoch()
        train_dataloader = create_train_dataloader(args, train_dataset)

        image_encoder.train()

        for batch in train_dataloader:

            # grid patcher returns 5-tuple (with padding masks); random/sift return 3-tuple
            if len(batch) == 5:
                images_1, images_2, labels, mask_1, mask_2 = batch
            else:
                images_1, images_2, labels = batch
                mask_1 = mask_2 = None

            embedding, loss_value = train_one_step(
                image_encoder=image_encoder,
                optimizer=optimizer,
                loss_optimizer=loss_optimizer,
                loss_object=loss_object,
                images_1=images_1,  # shape: [batch_size, patch_count, channels, height, width]
                images_2=images_2,
                labels=labels,
                device=device,
                scaler=scaler,
                mask_1=mask_1,
                mask_2=mask_2,
            )

            if loss_value:
                epoch_loss_sum += loss_value
                epoch_steps += 1

        # --- evaluation ---
        eval_start = time.time()
        image_encoder.eval()
        with torch.no_grad():
            metrics = test_identification(
                encoder=image_encoder,
                gallery_dataloader=gallery_dataloader,
                query_dataloader=query_dataloader,
                device=device,
            )
            logger.info(
                f"CSI METRICS epoch {epoch}: "
                + " ".join(
                    f"{k}:{v}"
                    for k, v in [
                        (
                            "cmc",
                            "[" + ", ".join(f"{x:.3f}" for x in metrics.csi_metrics.cmc[:10]) + "]",
                        ),
                        ("mrr", f"{metrics.csi_metrics.mrr:.3f}"),
                    ]
                )
            )
            logger.info(
                f"OSI METRICS epoch {epoch}:\n"
                + "\n".join(
                    f"{k}: {v}" for k, v in metrics.osi_metrics.main_fpir_op_points.items()
                )
            )

        time_end_epoch = time.time()
        epoch_time_sec = time_end_epoch - epoch_start
        epoch_time = convert_sec_to_hours_minutes_seconds(epoch_time_sec)
        eval_time_sec = time_end_epoch - eval_start
        eval_time = convert_sec_to_hours_minutes_seconds(eval_time_sec)
        train_time = convert_sec_to_hours_minutes_seconds(epoch_time_sec - eval_time_sec)
        avg_loss = epoch_loss_sum / max(epoch_steps, 1)
        logger.info(f"Epoch {epoch} | avg loss: {avg_loss:.4f} | time: {epoch_time} | train time: {train_time} | eval time: {eval_time}")

        if epoch >= args.epochs:
            break

    time_end = time.time()
    total_time = convert_sec_to_hours_minutes_seconds(time_end - time_start)
    setup_time = convert_sec_to_hours_minutes_seconds(time_setup_finished - time_start)
    train_eval_time = convert_sec_to_hours_minutes_seconds(time_end - time_setup_finished)

    logger.info(f"Training completed in {total_time} (setup time: {setup_time}, training+eval time: {train_eval_time})")


if __name__ == '__main__':
    main()
