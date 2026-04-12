import argparse
import logging
import os
import sys
import time

import torch
import torchvision
from torch.utils.data import DataLoader

from pytorch_metric_learning import losses

from src.model import WriterIdentificationEncoder
from src.id_dataset import IdDataset

from src.patchers.patcher_config import PatcherConfig, PATCH_METHODS
from src.patchers.collate import pad_patches_collate

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
        "--gt-file-tst",
        help="Testing text file with an image file name and id on each line."
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

    parser.add_argument("--width", type=int, default=320)

    parser.add_argument("--start-iteration", default=0, type=int)
    parser.add_argument("--max-iterations", default=50000, type=int)
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

    parser.add_argument("--eval-on-start", action="store_true")
    parser.add_argument("--logging-level", default="INFO")

    return parser.parse_args()


def configure_logging(logging_level: str) -> None:

    """
    Configure root logger.

    Parameters:
        logging_level (str): Logging level as string.
    """

    log_formatter = logging.Formatter(
        "CONVERT LINES TO JSONL - %(asctime)s - %(filename)s - %(levelname)s - %(message)s"
    )
    log_formatter.converter = time.gmtime

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(log_formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(handler)
    root_logger.setLevel(logging_level)


# def ensure_directories(args) -> tuple[str, str, str]:
#
#     """
#     Create output directories if they do not exist.
#
#     Parameters:
#         args (argparse.Namespace): Parsed arguments.
#
#     Returns:
#         tuple[str, str, str]: Paths for heat maps, t-SNE outputs, and retrieval outputs.
#     """
#
#     show_dir_heat_maps_path = os.path.join(args.show_dir, "heat_maps")
#     show_dir_tsne_path = os.path.join(args.show_dir, "tsne")
#     show_dir_retrieval_path = os.path.join(args.show_dir, "retrieval")
#
#     os.makedirs(show_dir_heat_maps_path, exist_ok=True)
#     os.makedirs(show_dir_tsne_path, exist_ok=True)
#     os.makedirs(show_dir_retrieval_path, exist_ok=True)
#     os.makedirs(args.out_checkpoints_dir, exist_ok=True)
#
#     return show_dir_heat_maps_path, show_dir_tsne_path, show_dir_retrieval_path


def create_model(args, device: torch.device) -> torch.nn.Module:

    """
    Create encoder model and optionally load checkpoint.

    Parameters:
        args (argparse.Namespace): Parsed arguments.
        device (torch.device): Device used for computation.

    Returns:
        torch.nn.Module: Initialized encoder model.
    """

    image_encoder = WriterIdentificationEncoder(
        in_channels=3,
        hidden_dim=256,
        embed_dim=args.embed_dim,
        nhead=8,
        num_transformer_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
        use_positional_encoding=(args.patcher == "grid"),
    ).to(device)

    if args.start_iteration > 0:
        checkpoint_path = os.path.join(
            args.out_checkpoints_dir,
            f"cp-{args.start_iteration:07d}.img.ckpt"
        )
        logger.info(f"Loading image checkpoint {checkpoint_path}")
        image_encoder.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )

    return image_encoder


def create_dataloaders(args) -> tuple[DataLoader, DataLoader | None]:

    """
    Create training and optional testing dataloaders.

    Parameters:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        tuple[DataLoader, DataLoader | None]: Training dataloader and testing dataloader.
    """

    transform = torchvision.transforms.ToTensor()

    # make patcher config
    patcher_config = PatcherConfig(
        method=args.patcher,
        patch_count=args.patch_count,
        random_seed=NP_RANDOM_SEED,
        patch_height=args.patch_height,
        patch_width=args.patch_width,
        min_partial_ratio=0.3,
    )

    # grid patcher produces variable patch counts per image, so a custom collate pads to the max count and creates masks
    collate_fn = pad_patches_collate if args.patcher == "grid" else None

    # training dataset uses augmentation
    train_dataset = IdDataset(
        args.gt_file,
        args.lmdb,
        transform=transform,
        augment=True,
        width=args.width,
        patcher_config=patcher_config
    )
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=8,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    test_dataloader = None

    if args.gt_file_tst:
        # testing dataset disables augmentation and uses deterministic cropping
        test_dataset = IdDataset(
            args.gt_file_tst,
            args.lmdb,
            transform=transform,
            augment=False,
            restrict_data=True,
            test=True,
            width=args.width,
            patcher_config=patcher_config
        )
        test_dataloader = DataLoader(
            test_dataset,
            num_workers=0,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn,
        )

    return train_dataloader, test_dataloader


# def save_first_batch_images(images_1: torch.Tensor, images_2: torch.Tensor, show_dir: str) -> None:
#
#     """
#     Save the first training batch as preview images
#
#     Parameters:
#         images_1 (torch.Tensor): First image batch.
#         images_2 (torch.Tensor): Second image batch.
#         show_dir (str): Output directory.
#     """
#
#     # convert tensors from (N, C, H, W) to (N, H, W, C) for OpenCV saving.
#     batch_image_1 = np.concatenate(
#         images_1.permute(0, 2, 3, 1).numpy(),
#         axis=0
#     ) * 255
#     batch_image_2 = np.concatenate(
#         images_2.permute(0, 2, 3, 1).numpy(),
#         axis=0
#     ) * 255
#
#     cv2.imwrite(os.path.join(show_dir, "images1.png"), batch_image_1)
#     cv2.imwrite(os.path.join(show_dir, "images2.png"), batch_image_2)


# def save_embedding_heatmap(embedding: torch.Tensor, output_path: str) -> None:
#
#     """
#     Save embedding similarity heatmap.
#
#     Parameters:
#         embedding (torch.Tensor): Batch embeddings.
#         output_path (str): Path to output image.
#     """
#
#     fig, ax = plt.subplots()
#
#     # dot product matrix roughly shows how similar embeddings in the batch are
#     distances = torch.mm(embedding, embedding.t()).detach().cpu().numpy()
#
#     heatmap = ax.imshow(distances)
#     plt.colorbar(heatmap)
#     plt.savefig(output_path)
#     plt.close("all")


def should_run_evaluation(iteration: int, args, start_iteration: int) -> bool:

    """
    Decide whether evaluation should be run

    Parameters:
        iteration (int): Current iteration.
        args (argparse.Namespace): Parsed arguments.
        start_iteration (int): Starting iteration.

    Returns:
        bool: True if evaluation should be run.
    """

    return (
        (iteration % args.view_step == 0 and iteration > start_iteration)
        or (args.eval_on_start and iteration == start_iteration)
    )


def train_one_step(
    image_encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_object,
    images_1: torch.Tensor,
    images_2: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
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
        loss_object: Metric learning loss object.
        images_1 (torch.Tensor): First batch of patched images.
        images_2 (torch.Tensor): Second batch of patched images.
        labels (torch.Tensor): Batch labels.
        device (torch.device): Computation device.
        mask_1 (torch.Tensor, optional, default=None): Padding mask for images_1. True = padded position to ignore.
            Used when the grid patcher produces variable patch counts per image.
        mask_2 (torch.Tensor, optional, default=None): Padding mask for images_2. Same convention as mask_1.

    Returns:
        tuple[torch.Tensor, float]: embeddings and scalar loss value.
    """

    # move images and labels to target device
    images_1 = images_1.to(device)
    images_2 = images_2.to(device)
    labels = labels.to(device)

    # concatenate both positive views into one batch
    images = torch.cat([images_1, images_2], dim=0)  # [2 * batch-size, patch-count, channels, height, width]
    labels = torch.cat([labels, labels], dim=0)      # [2 * batch-size]

    # concatenate padding masks the same way (None when all patches are real)
    padding_mask = None
    if mask_1 is not None and mask_2 is not None:
        padding_mask = torch.cat([mask_1, mask_2], dim=0).to(device)  # [2 * batch-size, patch-count]

    optimizer.zero_grad()

    embedding = image_encoder(images, padding_mask=padding_mask)  # [2 * batch-size, D]
    loss = loss_object(embedding, labels)

    loss.backward()
    optimizer.step()

    return embedding, loss.item()


# def evaluate_and_save_outputs(
#     iteration: int,
#     loss_history: list[float],
#     last_view_iteration: int,
#     t_start: float,
#     image_encoder: torch.nn.Module,
#     embedding: torch.Tensor,
#     args,
#     show_dir_heat_maps_path: str,
#     show_dir_tsne_path: str,
#     show_dir_retrieval_path: str,
#     test_dataloader: DataLoader | None,
#     device: torch.device
# ) -> int:
#
#     """
#     Run evaluation, logging, checkpoint saving, and visualization.
#
#     Parameters:
#         iteration (int): Current iteration.
#         loss_history (list[float]): Training loss history.
#         last_view_iteration (int): Last evaluation iteration.
#         t_start (float): Evaluation timer start.
#         image_encoder (torch.nn.Module): Encoder model.
#         embedding (torch.Tensor): Current batch embeddings.
#         args (argparse.Namespace): Parsed arguments.
#         show_dir_heat_maps_path (str): Heat map output directory.
#         show_dir_tsne_path (str): t-SNE output directory.
#         show_dir_retrieval_path (str): Retrieval output directory.
#         test_dataloader (DataLoader | None): Optional testing dataloader.
#         device (torch.device): Computation device.
#
#     Returns:
#         int: Updated last_view_iteration.
#     """
#
#     avg_loss = np.mean(loss_history[last_view_iteration:])
#
#     logger.info(
#         f"LOG {iteration} iterations:{iteration - last_view_iteration} "
#         f"loss:{avg_loss:0.6f} time:{time.time() - t_start:.1f}s"
#     )
#
#     checkpoint_path = os.path.join(
#         args.out_checkpoints_dir,
#         f"cp-{iteration:07d}.img.ckpt"
#     )
#     torch.save(image_encoder.state_dict(), checkpoint_path)
#
#     heatmap_path = os.path.join(
#         show_dir_heat_maps_path,
#         f"cp-{iteration:07d}.png"
#     )
#     save_embedding_heatmap(embedding, heatmap_path)
#
#     if test_dataloader is not None:
#
#         retrieval_output_path = os.path.join(
#             show_dir_retrieval_path,
#             f"retrieval-{iteration:07d}.{os.path.basename(args.gt_file_tst)}.png"
#         )
#
#         auc, mean_auc, fpr, tpr, thr, mean_ap = test_retrieval(
#             retrieval_output_path,
#             image_encoder,
#             test_dataloader,
#             device
#         )
#
#         logger.info(
#             f"TEST {iteration} AUC:{auc:0.6f} "
#             f"MEAN_AUC:{mean_auc:0.6f} MEAN_AP:{mean_ap:0.6f}"
#         )
#
#         tsne_output_path = os.path.join(
#             show_dir_tsne_path,
#             f"tsne-{iteration:07d}.tst.png"
#         )
#         plot_tsne(
#             tsne_output_path,
#             image_encoder,
#             test_dataloader,
#             device,
#             logger
#         )
#
#     return iteration
#
#
# def test_retrieval(
#     file_name: str,
#     image_encoder: torch.nn.Module,
#     dataloader: DataLoader,
#     device: torch.device,
#     query_vis_count: int = 32,
#     result_vis_count: int = 20
# ) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, float]:
#     """
#     Evaluate retrieval quality using embedding similarities
#
#     Parameters:
#         file_name (str): Output image path for retrieval collage
#         image_encoder (torch.nn.Module): Encoder model
#         dataloader (DataLoader): Testing dataloader
#         device (torch.device): Computation device
#         query_vis_count (int): Number of query images visualized
#         result_vis_count (int): Number of retrieval results per query.
#
#     Returns:
#         tuple:
#             auc (float): Global ROC AUC.
#             mean_auc (float): Mean per-query AUC.
#             fpr (np.ndarray): False positive rates.
#             tpr (np.ndarray): True positive rates.
#             thr (np.ndarray): Thresholds.
#             mean_ap (float): Mean average precision.
#     """
#
#     t_start = time.time()
#
#     all_embeddings = []
#     all_labels = []
#     all_images = []
#
#     with torch.no_grad():
#
#         for images_1, images_2, labels in dataloader:
#             # retrieval uses only the first image branch.
#             images_1 = images_1.to(device)
#             embedding = image_encoder(images_1)
#
#             all_embeddings.append(embedding.cpu().numpy())
#             all_labels.append(labels.cpu().numpy())
#             all_images.append(images_1.permute(0, 2, 3, 1).cpu().numpy())
#
#     all_embeddings = np.concatenate(all_embeddings, axis=0)
#     all_labels = np.concatenate(all_labels, axis=0)
#     all_images = np.concatenate(all_images, axis=0) * 255
#
#     # similarity matrix between all test embeddings
#     similarities = np.dot(all_embeddings, all_embeddings.T)
#
#     logging.info(
#         f"SIMILARITIES {similarities.min()} {similarities.max()}, {similarities.shape}"
#     )
#
#     collage_ids = np.linspace(
#         0,
#         all_images.shape[0] - 1,
#         query_vis_count
#     ).astype(np.int64)
#
#     result_collage = []
#
#     scores = []
#     labels = []
#     ap = []
#     auc = []
#
#     for query_index in range(similarities.shape[0]):
#
#         query_similarity = similarities[query_index]
#         query_labels = all_labels[query_index] == all_labels
#
#         # remove self-match from evaluation.
#         query_labels[query_index] = False
#         query_similarity[query_index] = -1e20
#
#         if np.any(query_labels):
#             auc.append(roc_auc_score(query_labels, query_similarity))
#             ap.append(average_precision_score(query_labels, query_similarity))
#
#         scores.append(query_similarity)
#         labels.append(query_labels)
#
#         if query_index in collage_ids:
#             # blue border marks the query image
#
#             query_image = cv2.copyMakeBorder(
#                 all_images[query_index],
#                 2, 2, 2, 2,
#                 cv2.BORDER_CONSTANT,
#                 value=[255, 0, 0]
#             )
#             query_image = cv2.copyMakeBorder(
#                 query_image,
#                 1, 1, 1, 1,
#                 cv2.BORDER_CONSTANT,
#                 value=[0, 0, 0]
#             )
#
#             row = [query_image]
#             query_label = all_labels[query_index]
#
#             best_ids = np.argsort(query_similarity)[::-1][:result_vis_count]
#
#             for result_index in best_ids:
#                 result_image = all_images[result_index]
#
#                 # green border means correct retrieval, red means wrong retrieval.
#                 if all_labels[result_index] == query_label:
#                     result_image = cv2.copyMakeBorder(
#                         result_image,
#                         2, 2, 2, 2,
#                         cv2.BORDER_CONSTANT,
#                         value=[0, 255, 0]
#                     )
#                 else:
#                     result_image = cv2.copyMakeBorder(
#                         result_image,
#                         2, 2, 2, 2,
#                         cv2.BORDER_CONSTANT,
#                         value=[0, 0, 255]
#                     )
#
#                 result_image = cv2.copyMakeBorder(
#                     result_image,
#                     1, 1, 1, 1,
#                     cv2.BORDER_CONSTANT,
#                     value=[0, 0, 0]
#                 )
#                 row.append(result_image)
#
#             result_collage.append(np.concatenate(row, axis=1))
#
#     if result_collage:
#         result_collage = np.concatenate(result_collage, axis=0)
#         cv2.imwrite(file_name, result_collage)
#
#     scores = np.concatenate(scores, axis=0)
#     labels = np.concatenate(np.asarray(labels), axis=0)
#
#     fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
#     mean_ap = np.mean(ap)
#     mean_auc = np.mean(auc)
#     auc = roc_auc_score(labels, scores)
#
#     logger.info(f"TEST TIME: {time.time() - t_start}s")
#
#     return auc, mean_auc, fpr, tpr, thr, mean_ap


def main() -> None:

    """
    Main training entry point.
    """

    args = parse_args()

    configure_logging(args.logging_level)

    logger.info(" ".join(sys.argv))
    logger.info(f"ARGS {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # spawn start method is often safer when DataLoader uses workers.
    torch.multiprocessing.set_start_method("spawn")

    # TO DO: Ensuring output directories (for now being skipped - don't know what directories to create and for what)
    # show_dir_heat_maps_path, show_dir_tsne_path, show_dir_retrieval_path = ensure_directories(args)

    image_encoder = create_model(args, device)
    train_dataloader, test_dataloader = create_dataloaders(args)

    optimizer = torch.optim.AdamW(
        image_encoder.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    loss_history = [0] * args.start_iteration
    iteration = args.start_iteration
    last_view_iteration = args.start_iteration

    loss_object = losses.NTXentLoss(temperature=args.temperature)

    epoch = 0

    while True:
        epoch += 1
        epoch_start = time.time()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for batch in train_dataloader:

            # grid patcher returns 5-tuple (with padding masks), random/sift return 3-tuple
            if len(batch) == 5:
                images_1, images_2, labels, mask_1, mask_2 = batch
            else:
                images_1, images_2, labels = batch
                mask_1 = mask_2 = None

            # if iteration == args.start_iteration:
            #     # save preview of the first batch for quick visual inspection.
            #     save_first_batch_images(images_1, images_2, args.show_dir)

            embedding, loss_value = train_one_step(
                image_encoder=image_encoder,
                optimizer=optimizer,
                loss_object=loss_object,
                images_1=images_1,  # shape: [batch_size, patch_count, channels, height, width]
                images_2=images_2,
                labels=labels,
                device=device,
                mask_1=mask_1,
                mask_2=mask_2,
            )
            loss_history.append(loss_value)
            epoch_loss_sum += loss_value
            epoch_steps += 1

            # This validation method is being skipped (not valid for our method of training)
            if should_run_evaluation(iteration, args, args.start_iteration):
                pass
            #     last_view_iteration = evaluate_and_save_outputs(
            #         iteration=iteration,
            #         loss_history=loss_history,
            #         last_view_iteration=last_view_iteration,
            #         t_start=evaluation_timer_start,
            #         image_encoder=image_encoder,
            #         embedding=embedding,
            #         args=args,
            #         show_dir_heat_maps_path=show_dir_heat_maps_path,
            #         show_dir_tsne_path=show_dir_tsne_path,
            #         show_dir_retrieval_path=show_dir_retrieval_path,
            #         test_dataloader=test_dataloader,
            #         device=device
            #     )
            #     evaluation_timer_start = time.time()

            iteration += 1

            if iteration >= args.max_iterations:
                break

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss_sum / max(epoch_steps, 1)
        logger.info(f"Epoch {epoch} | iter {iteration}/{args.max_iterations} | avg loss: {avg_loss:.4f} | time: {epoch_time:.1f}s")

        if iteration >= args.max_iterations:
            break


if __name__ == '__main__':
    main()
