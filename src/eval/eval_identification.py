import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from .eval_csi import _calc_CSI_metrics
from .eval_osi import _calc_OSI_metrics
from .metrics_identification import IdentificationMetrics
from .eval_identification_ranking import _get_gallery_ranking, _pool_gallery_to_class_ranking


def _get_all_from_dataloader(encoder: torch.nn.Module,
                             dataloader: DataLoader,
                             device: torch.device):
    """
    Get all image samples, author labels and embeddings computed by the model for each sample.

    Parameters:
        encoder (torch.nn.Module): Encoder to compute embeddings.

        dataloader (DataLoader): Dataloader to load the samples.

        device (torch.device): Device used to run the model inference.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            all_labels (np.ndarray, shape=(N_samples,), dtype=int): Array of labels.

            all_embeds (np.ndarray, shape=(N_samples, Embed_size), dtype=np.float32): Array of embeddings.
    """

    all_embeds = []
    # all_images = []
    all_labels = []

    with torch.no_grad():
        # TODO: Need to modify to work with newest dataloader implementation

        for batch in dataloader:
            # grid patcher returns 5-tuple (with padding masks), random/sift return 3-tuple
            if len(batch) == 5:
                samples_1, samples_2, labels, mask_1, mask_2 = batch
            else:
                samples_1, samples_2, labels = batch
                mask_1 = None

            # (B, P, C, H, W)
            samples_1 = samples_1.to(device)
            padding_mask = mask_1.to(device) if mask_1 is not None else None
            embeds = encoder(samples_1, padding_mask=padding_mask)

            # # (B, P, C, H, W) -> (B, P, H, W, C)
            # # multiplying by 255 to scale values from [0, 1] to [0, 255]
            # images_np = images_1.permute(
            #     0, 1, 3, 4, 2).cpu().numpy().astype(np.float32) * 255.0

            # (B)
            labels_np = labels.cpu().numpy().astype(np.float32)

            # (B, Embed_size)
            embeds_np = embeds.cpu().numpy().astype(np.float32)

            # all_images.append(images_np)
            all_labels.append(labels_np)
            all_embeds.append(embeds_np)

    # # (N_samples, H, W, C)
    # all_images = np.concatenate(all_images, axis=0)
    # (N_samples)
    all_labels = np.concatenate(all_labels, axis=0)
    # (N_samples, Embed_size)
    all_embeds = np.concatenate(all_embeds, axis=0)

    # sort by labels
    order = np.argsort(all_labels)
    # all_images = all_images[order]
    all_labels = all_labels[order]
    all_embeds = all_embeds[order]

    return all_labels, all_embeds


def _get_prototypes(gallery_labels: np.ndarray,
                    gallery_embeds: np.ndarray):
    """
    Create prototypes from gallery samples.
    Prototype embedding for specific label is computed as average of gallery embeddings with the same label.

    Parameters:
        gallery_labels (np.ndarray, shape=(N_samples), dtype=int):
            Array of gallery labels.

        gallery_embeds (np.ndarray, shape=(N_samples, Embed_size), dtype=np.float32):
            Array of gallery embeddings.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            proto_labels (np.ndarray, shape=(N_labels,), dtype=int): Array of prototype labels.

            proto_embeds (np.ndarray, shape=(N_labels, Embed_size), dtype=np.float32): Array of prototype embeddings.
    """

    # gallery labels are already sorted (and also are the corresponding embeddings)
    # so np.unique will return start indices of the same label segments
    # (N_labels), (N_labels)
    proto_labels, start_idxs = np.unique(gallery_labels, return_index=True)

    N_samples = len(gallery_labels)

    # proto_images = []
    proto_embeds = []

    for i in range(len(proto_labels)):
        start = start_idxs[i]
        if i + 1 < len(proto_labels):
            end = start_idxs[i + 1]
        else:
            end = N_samples

        proto_embeds.append(gallery_embeds[start:end].mean(axis=0))

        # get first image as prototype image
        # proto_images.append(gallery_images[start])

    # (N_labels, Embed_size)
    proto_embeds = np.stack(proto_embeds)

    # (N_labels, H, W, C)
    # proto_images = np.stack(proto_images)

    # normalize prototype embeddings
    proto_embeds = proto_embeds / \
        np.linalg.norm(proto_embeds, axis=1, keepdims=True)

    # (N_labels,), (N_labels, Embed_size)
    return proto_labels, proto_embeds


def eval_identification(
        encoder: torch.nn.Module,
        gallery_dataloader: DataLoader,
        query_dataloader: DataLoader,
        device: torch.device
):
    """
    Evaluates encoder on closed-set and open-set identification task.

    Parameters:
        encoder (torch.nn.Module): Encoder, that encodes samples into class embeddings.
        gallery_dataloader (DataLoader): Loads gallery samples.
        query_dataloader (DataLoader): Loads query samples.
        device (torch.device): Device on which the encoder will be run.

    Returns:
        IdentificationMetrics: Computed closed-set and open-set identification metrics.
    """

    start_time = time.time()

    # load all gallery samples
    gallery_labels, gallery_embeds = _get_all_from_dataloader(
        encoder, gallery_dataloader, device)

    # load all query samples
    query_labels, query_embeds = _get_all_from_dataloader(
        encoder, query_dataloader, device)

    # normalize embeddings to lie on unit hypersphere (in case they were not normalized)
    gallery_embeds = gallery_embeds / \
        np.linalg.norm(gallery_embeds, axis=1, keepdims=True)
    query_embeds = query_embeds / \
        np.linalg.norm(query_embeds, axis=1, keepdims=True)

    gallery_labels_ranked, gallery_scores_ranked = _get_gallery_ranking(
        gallery_labels, gallery_embeds, query_embeds)

    class_labels_ranked, class_scores_ranked = _pool_gallery_to_class_ranking(
        gallery_labels_ranked, gallery_scores_ranked)

    csi_metrics = _calc_CSI_metrics(
        gallery_labels_ranked, class_labels_ranked, query_labels)

    osi_metrics = _calc_OSI_metrics(
        class_labels_ranked, class_scores_ranked, query_labels)

    eval_time = time.time() - start_time

    return IdentificationMetrics(csi_metrics=csi_metrics,
                                 osi_metrics=osi_metrics,
                                 eval_time=eval_time)
