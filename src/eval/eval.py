import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from calc_csi_metrics import _calc_CSI_metrics
from calc_osi_metrics import _calc_OSI_metrics
from metrics import IdnetificationMetrics


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

        # (B, P, C, H, W), (B, P, C, H, W), (B,)
        for samples_1, samples_2, labels in dataloader:
            # ignore second samples

            # (B, P, C, H, W)
            samples_1 = samples_1.to(device)
            embeds = encoder(samples_1)

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


def _get_gallery_ranking(gallery_labels: np.ndarray,
                         gallery_embeds: np.ndarray,
                         query_embeds: np.ndarray):
    """
    Parameters:
        gallery_labels (np.ndarray, shape=(N_gallery), dtype=int):
            Labels of all gallery samples.
        gallery_embeds (np.ndarray, shape=(N_gallery, Embed_size), dtype=np.float32):
            Embeddings of all gallery samples.
        query_embeds (np.ndarray, shape=(N_gallery, Embed_size), dtype=np.float32):
            Embeddings of all query samples.

    """

    gallery_idxs_ranked = []
    gallery_scores_ranked = []

    # iterate over batches of queries
    query_batch_size = 1024
    for i in range(0, len(query_embeds), query_batch_size):

        # select batch of queries
        # (query_batch_size, emb_size)
        query_batch = query_embeds[i:i+query_batch_size]

        # computing cosine similarity between query and gallery samples
        #   (gallery and query embeddings are already normalized)
        # (query_batch_size, emb_size) @ (N_gallery, emb_size)^T -> (query_batch_size, N_gallery)
        sims_batch = query_batch @ gallery_embeds.T

        # get indices that sort similarity scores from largest to smallest for each query
        # (query_batch_size, N_gallery)
        idxs = np.argsort(-sims_batch, axis=1)

        # sort the scores
        # (query_batch_size, N_gallery)
        scores = np.take_along_axis(sims_batch, idxs, axis=1)

        gallery_idxs_ranked.append(idxs)
        gallery_scores_ranked.append(scores)

    # stack batches into single array
    # list[(query_batch_size, N_gallery)] -> (N_queries, N_gallery)
    gallery_idxs_ranked = np.vstack(gallery_idxs_ranked)
    # list[(query_batch_size, N_gallery)] -> (N_queries, N_gallery)
    gallery_scores_ranked = np.vstack(gallery_scores_ranked)

    # map indices to actual labels
    # (N_queries, N_gallery)
    gallery_labels_ranked = gallery_labels[gallery_idxs_ranked]

    # (N_queries, N_gallery), (N_queries, N_gallery)
    return gallery_labels_ranked, gallery_scores_ranked


def _pool_gallery_to_class_ranking(gallery_labels_ranked: np.ndarray,
                                   gallery_scores_ranked: np.ndarray):

    N_queries = gallery_scores_ranked.shape[0]
    unique_labels = np.unique(gallery_labels_ranked)
    N_labels = len(unique_labels)

    # each row will contain max pooled score for each label in order of unique_labels
    class_scores = np.full((N_queries, N_labels), -np.inf)

    for i, label in enumerate(unique_labels):
        # create mask for this label across all queries
        mask = (gallery_labels_ranked == label)

        # max pool scores where mask is True
        class_scores[:, i] = np.where(
            # check if the label appears in the row at least once
            mask.any(axis=1),
            np.max(
                np.where(mask, gallery_scores_ranked, -np.inf),
                axis=1),  # max score for the current label row-wise
            -np.inf
        )

    # get labels corresponding to the scores
    # simply repeat row-wise the unique labels as scores are in this order
    # (N_queries, N_labels)
    class_labels = np.tile(unique_labels, (N_queries, 1))

    # get indexes that sort scores in each row in descending order
    # (N_queries, N_labels)
    score_order = np.argsort(-class_scores, axis=1)

    # sort the scores and corresponding lables
    class_scores_ranked = np.take_along_axis(
        class_scores, score_order, axis=1)
    class_labels_ranked = np.take_along_axis(
        class_labels, score_order, axis=1)

    return class_labels_ranked, class_scores_ranked


def test_identification(
        encoder: torch.nn.Module,
        gallery_dataloader: DataLoader,
        query_dataloader: DataLoader,
        device: torch.device
):
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

    # # get prototypes
    # # (N_labels,), (N_labels, emb_size)
    # proto_labels, proto_embeds = _get_prototypes(
    #     gallery_labels, gallery_embeds)

    # # get ranked gallery labels (authors) for each query
    # # (based on cosine similarity between embeddings)
    # # (N_queries, N_labels), (N_queries, N_labels)
    # full_ranking_labels, full_ranking_scores = _get_full_ranking(
    #     proto_labels, proto_embeds, query_embeds)

    gallery_labels_ranked, gallery_scores_ranked = _get_gallery_ranking(
        gallery_labels, gallery_embeds, query_embeds)

    class_labels_ranked, class_scores_ranked = _pool_gallery_to_class_ranking(
        gallery_labels_ranked, gallery_scores_ranked)

    csi_metrics = _calc_CSI_metrics(
        gallery_labels_ranked, class_labels_ranked, query_labels)

    osi_metrics = _calc_OSI_metrics(
        class_labels_ranked, class_scores_ranked, query_labels)

    eval_time = time.time() - start_time

    return IdnetificationMetrics(csi_metrics=csi_metrics,
                                 osi_metrics=osi_metrics,
                                 eval_time=eval_time)
