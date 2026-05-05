import argparse
import json
import logging
import os
import sys
import time

import numpy as np

from src.eval import IdentificationMetrics

np.bool = bool

import torch
from torch.utils.data import DataLoader

from pytorch_metric_learning import losses

from src.model import WriterIdentificationEncoder
from src.id_dataset import IdDataset, AuthorStratifiedBatchSampler
from src.utils import convert_sec_to_hours_minutes_seconds

from src.patchers.patcher_config import PatcherConfig, PATCH_METHODS
from src.patchers.collate import pad_patches_collate
from src.eval import eval_identification

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

    parser.add_argument(
        "--sift-keypoints-lmdb",
        type=str,
        default=None,
        help=(
            "Path to an LMDB produced by `python -m src.patchers.extract_keypoints` "
            "holding pre-computed SIFT keypoints. Required when --patcher sift."
        ),
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
    parser.add_argument("--max-stale-epochs", default=5, type=int, help="Maximum number of consecutive epochs without improvement before stopping training.")
    parser.add_argument("--out-checkpoints-dir", default='.', type=str)
    parser.add_argument("--out-model-name", default='knn_model', type=str)
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

    args = parser.parse_args()

    if args.patcher == "sift" and args.sift_keypoints_lmdb is None:
        parser.error("--sift-keypoints-lmdb is required when --patcher is 'sift'.")

    return args

def set_model_args(model_args: argparse.Namespace, checkpoint_args: argparse.Namespace) -> None:
    """
    Update model_args with values from checkpoint_args for keys that are relevant to the model.

    This is used when resuming training from a checkpoint to ensure that the model is configured
    consistently with the original training run, even if some arguments (like learning rate) are
    overridden by command-line parameters.

    Parameters:
        model_args (argparse.Namespace): The current model arguments that may have been overridden by command-line inputs.
        checkpoint_args (argparse.Namespace): The arguments loaded from the checkpoint, which contain the original training configuration.
    """
    # Define which argument keys are relevant for the model configuration
    relevant_keys = {
        "patcher",
        "patch_count",
        "patch_height",
        "patch_width",
        "embed_dim",
        "learning_rate",
        "weight_decay",
        "temperature",
        "samples_per_author",
        "min_authors_per_batch",
        "gt_file",
        "gt_file_gallery",
        "gt_file_query",
    }

    for key in relevant_keys:
        if hasattr(checkpoint_args, key):
            setattr(model_args, key, getattr(checkpoint_args, key))

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

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_optimizer: torch.optim.Optimizer, epoch: int, metrics: IdentificationMetrics, stop_counter: int, args) -> None:
    """
    Save model checkpoint.

    Parameters:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer whose state to save.
        loss_optimizer (torch.optim.Optimizer): Loss optimizer whose state to save.
        epoch (int): Current epoch number, used for naming the checkpoint file.
        metrics (dict): Evaluation metrics to save alongside the checkpoint.
        stop_counter (int): Number of consecutive epochs without improvement, saved for potential use in resuming training.
        args (argparse.Namespace): Parsed arguments containing output directory and model name.
    """
    metrics_json = metrics.to_json_compact() if metrics else None
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_optimizer_state_dict": loss_optimizer.state_dict(),
        "epoch": epoch,
        "stop_counter": stop_counter,
        "mAP": metrics.csi_metrics.mAP if metrics else None,
        "args": dict(vars(args)),
    }
    checkpoint_dir = os.path.join(args.out_checkpoints_dir, f"{args.out_model_name}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    args_path = os.path.join(checkpoint_dir, f"{args.out_model_name}_args.json")
    checkpoint_path = os.path.join(checkpoint_dir, f"{args.out_model_name}.img.ckpt")
    metrics_path = os.path.join(checkpoint_dir, f"{args.out_model_name}_metrics.json")
    tmp_path = checkpoint_path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, checkpoint_path)
    with open(metrics_path, "w") as f:
        f.write(metrics_json if metrics else "{}")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"Saved checkpoint: {checkpoint_dir}")

def load_args_from_checkpoint(checkpoint_path: str, device: torch.device) -> argparse.Namespace:
    """
    Load arguments from a checkpoint file.

    Parameters:
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Device to map the loaded checkpoint to.
    Returns:
        argparse.Namespace: Arguments loaded from the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_dict = checkpoint.get("args", {})
    return argparse.Namespace(**args_dict)

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_optimizer: torch.optim.Optimizer, checkpoint_path: str, device: torch.device) -> tuple[int, int, float]:
    """
    Load model checkpoint.

    Parameters:
        model (torch.nn.Module): Model to load state into.
        optimizer (torch.optim.Optimizer): Optimizer to load state into.
        loss_optimizer (torch.optim.Optimizer): Loss optimizer to load state into.
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Device to map the loaded checkpoint to.

    Returns:
        int: The epoch number from which the checkpoint was loaded.
        int: The stop counter value from the checkpoint, indicating how many consecutive epochs without improvement have occurred.
        float: The mAP value from the checkpoint, representing the best evaluation metric achieved up to that point.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mAP = checkpoint.get("mAP", 0.0)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loss_optimizer.load_state_dict(checkpoint["loss_optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    stop_counter = checkpoint.get("stop_counter", 0)
    logger.info(f"Loaded checkpoint from epoch {epoch}")
    return epoch, stop_counter, mAP

def checkpoint_exists(args) -> bool:
    """
    Check if a checkpoint file exists for the given arguments.

    Parameters:
        args (argparse.Namespace): Parsed arguments containing output directory and model name.

    Returns:
        bool: True if the checkpoint file exists, False otherwise.
    """
    checkpoint_dir = os.path.join(args.out_checkpoints_dir, f"{args.out_model_name}")
    checkpoint_path = os.path.join(checkpoint_dir, f"{args.out_model_name}.img.ckpt")
    return os.path.isfile(checkpoint_path)


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
        sift_keypoints_lmdb_path=args.sift_keypoints_lmdb,
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
        sift_keypoints_lmdb_path=args.sift_keypoints_lmdb,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    checkpoint_dir = os.path.join(args.out_checkpoints_dir, f"{args.out_model_name}")
    if checkpoint_exists(args):
        checkpoint_args = load_args_from_checkpoint(
            checkpoint_path=os.path.join(checkpoint_dir, f"{args.out_model_name}.img.ckpt"),
            device=device,
        )
        set_model_args(args, checkpoint_args)

    configure_logging(args.logging_level)

    log_args(args, logger)

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

    best_mAP = 0.0
    stop_counter = 0

    if checkpoint_exists(args):
        checkpoint_dir = os.path.join(args.out_checkpoints_dir, f"{args.out_model_name}")
        epoch, stop_counter, best_mAP = load_checkpoint(
            model=image_encoder,
            optimizer=optimizer,
            loss_optimizer=loss_optimizer,
            checkpoint_path=os.path.join(checkpoint_dir, f"{args.out_model_name}.img.ckpt"),
            device=device,
        )
        logger.info(f"Resuming training from epoch {epoch} with previous best mAP {best_mAP:.4f}")

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
        epoch_metrics = None
        with torch.no_grad():
            epoch_metrics = eval_identification(
                encoder=image_encoder,
                gallery_dataloader=gallery_dataloader,
                query_dataloader=query_dataloader,
                device=device,
            )
            logger.info(epoch_metrics.to_json_compact())

        time_end_epoch = time.time()
        epoch_time_sec = time_end_epoch - epoch_start
        epoch_time = convert_sec_to_hours_minutes_seconds(epoch_time_sec)
        eval_time_sec = time_end_epoch - eval_start
        eval_time = convert_sec_to_hours_minutes_seconds(eval_time_sec)
        train_time = convert_sec_to_hours_minutes_seconds(epoch_time_sec - eval_time_sec)
        avg_loss = epoch_loss_sum / max(epoch_steps, 1)
        logger.info(f"Epoch {epoch} | avg loss: {avg_loss:.4f} | time: {epoch_time} | train time: {train_time} | eval time: {eval_time}")

        if epoch_metrics.csi_metrics.mAP > best_mAP:
            stop_counter = 0
            best_mAP = epoch_metrics.csi_metrics.mAP
            save_checkpoint(
                model=image_encoder,
                optimizer=optimizer,
                loss_optimizer=loss_optimizer,
                epoch=epoch,
                metrics=epoch_metrics,
                stop_counter=stop_counter,
                args=args,
            )
        else:
            stop_counter += 1

        if epoch >= args.epochs or stop_counter >= args.max_stale_epochs:
            if stop_counter >= args.max_stale_epochs:
                logger.info(f"No improvement in mAP for {stop_counter} consecutive epochs. Stopping training.")
            logger.info(f"Exiting training loop. Epoch {epoch} completed. Best mAP: {best_mAP:.4f}.")
            break

    time_end = time.time()
    total_time = convert_sec_to_hours_minutes_seconds(time_end - time_start)
    setup_time = convert_sec_to_hours_minutes_seconds(time_setup_finished - time_start)
    train_eval_time = convert_sec_to_hours_minutes_seconds(time_end - time_setup_finished)

    logger.info(f"Training completed in {total_time} (setup time: {setup_time}, training+eval time: {train_eval_time})")


if __name__ == '__main__':
    main()
