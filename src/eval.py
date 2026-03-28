import logging
import time

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

# from collections import defaultdict

from id_dataset import logger


def _get_all_from_dataloader(encoder: torch.nn.Module, dataloader: DataLoader, device: torch.device):
    all_embeds = []
    all_images = []
    all_labels = []

    with torch.no_grad():
        # TODO: Need to modify to work with newest dataloader implementation

        for images_1, images_2, labels in dataloader:
            # ignore second image

            images_1 = images_1.to(device)
            embeds = encoder(images_1)

            # (B, C, H, W) -> (B, H, W, C)
            # multiplying by 255 to scale values from [0, 1] to [0, 255]
            images_np = images_1.permute(
                0, 2, 3, 1).cpu().numpy().astype(np.float32) * 255.0

            # (B)
            labels_np = labels.cpu().numpy().astype(np.float32)

            # (B, Embedding size)
            embeds_np = embeds.cpu().numpy().astype(np.float32)

            all_images.append(images_np)
            all_labels.append(labels_np)
            all_embeds.append(embeds_np)

    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_embeds = np.concatenate(all_embeds, axis=0)

    # sort by labels
    perm = np.argsort(all_labels)
    all_images = all_images[perm]
    all_labels = all_labels[perm]
    all_embeds = all_embeds[perm]

    return all_images, all_labels, all_embeds


def _get_prototypes(gallery_images: np.ndarray, gallery_labels: np.ndarray, gallery_embeds: np.ndarray):
    # gallery labels are already sorted (and also are the corresponding embeddings)
    # so np.unique will return start indices of the same label segments
    # (N_authors), (N_authors)
    proto_labels, start_idxs = np.unique(gallery_labels, return_index=True)

    proto_images = []
    proto_embeds = []

    for i in range(len(proto_labels)):
        start = start_idxs[i]
        if i + 1 < len(proto_labels):
            end = start_idxs[i + 1]
        else:
            end = len(proto_labels)
        proto_embeds.append(gallery_embeds[start:end].mean(axis=0))
        proto_images.append(gallery_images[start])

    # (N_authors, emb_size)
    proto_embeds = np.stack(proto_embeds)

    # (N_authors, H, W, C)
    proto_images = np.stack(proto_images)

    # normalize prototype embeddings
    proto_embeds = proto_embeds / \
        np.linalg.norm(proto_embeds, axis=1, keepdims=True)

    # (N_authors, H, W, C), (N_authors), (N_authors, emb_size)
    return proto_images, proto_labels, proto_embeds


def _get_full_ranking(proto_labels: np.ndarray, proto_embeds: np.ndarray, quary_embeds: np.ndarray):

    query_batch_size = 1024

    full_ranking_idxs = []
    full_ranking_scores = []

    for i in range(0, len(quary_embeds), query_batch_size):

        # (query_batch_size, emb_size)
        query_batch = quary_embeds[i:i+query_batch_size]

        # (query_batch_size, emb_size) @ (N_labels, emb_size)^T -> (query_batch_size, N_labels)
        sims_batch = query_batch @ proto_embeds.T

        # (query_batch_size, N_labels)
        idxs = np.argsort(-sims_batch, axis=1)

        # (query_batch_size, N_labels)
        scores = np.take_along_axis(sims_batch, idxs, axis=1)

        full_ranking_idxs.append(idxs)
        full_ranking_scores.append(scores)

    # list[(query_batch_size, N_labels)] -> (N_queries, N_labels)
    full_ranking_idxs = np.vstack(full_ranking_idxs)
    # list[(query_batch_size, N_labels)] -> (N_queries, N_labels)
    full_ranking_scores = np.vstack(full_ranking_scores)

    # map indices to actual author ID
    # (N_queries, N_labels)
    full_ranking_labels = proto_labels[full_ranking_idxs]

    # (N_queries, N_labels), (N_queries, N_labels)
    return full_ranking_labels, full_ranking_scores


def _measure_rank_k_acc(full_ranking_labels: np.ndarray, query_labels: np.ndarray, k=1):
    correct = np.any(full_ranking_labels[:, :k] == query_labels[:, None], axis=1)
    return np.mean(correct)


def _measure_cmc(full_ranking_labels: np.ndarray, query_labels: np.ndarray):
    # measure CMC (Cumulative Match Characteristic)

    # create boolean matrix, on each row set a cell to True,
    # if the ranked label for the query matches the query label
    # (there should be only one match for each query)
    # (N_queries, N_labels)
    matches = (full_ranking_labels == query_labels[:, None])

    # find index of (first and only) label match
    # (N_queries)
    first_correct_idx = np.argmax(matches, axis=1)

    n_labels = len(full_ranking_labels)
    cmc_curve = np.zeros(n_labels)

    for k in range(n_labels):
        # check if first correct label was within rank k for each query
        # then calculate mean of this boolean array
        cmc_curve[k] = np.mean(first_correct_idx <= k)

    return cmc_curve


def _measure_mrr(full_ranking_labels: np.ndarray, query_labels: np.ndarray):
    # measure MRR (Mean Reciprocal Rank)

    # create boolean matrix, on each row set a cell to True,
    # if the ranked label for the query matches the query label
    # (there should be only one match for each query)
    # (N_queries, N_labels)
    matches = (full_ranking_labels == query_labels[:, None])

    # find index of (first and only) label match
    # (N_queries)
    first_correct_idx = np.argmax(matches, axis=1)

    # calculate mean of (1 / (first correct label index + 1))
    mrr = np.mean(1 / (first_correct_idx + 1))

    return mrr


def test_identif_closed_set(
        encoder: torch.nn.Module,
        gallery_dataloader: DataLoader,
        query_dataloader: DataLoader,
        device: torch.device
):
    start_time = time.time()

    # load all gallery and query sampels using corresponding dataloader
    gallery_images, gallery_labels, gallery_embeds = _get_all_from_dataloader(
        encoder, gallery_dataloader, device)
    query_images, query_labels, query_embeds = _get_all_from_dataloader(
        encoder, query_dataloader, device)

    # normalize embeddings to lie on unit hypersphere (in case they were not normalized)
    gallery_embeds = gallery_embeds / \
        np.linalg.norm(gallery_embeds, axis=1, keepdims=True)
    query_embeds = query_embeds / \
        np.linalg.norm(query_embeds, axis=1, keepdims=True)

    # get prototypes
    # (N_labels, H, W, C), (N_labels), (N_labels, emb_size)
    proto_images, proto_labels, proto_embeds = _get_prototypes(gallery_images, gallery_labels, gallery_embeds)

    # get ranked gallery labels (authors) for each query
    # (N_queries, N_labels), (N_queries, N_labels)
    full_ranking_labels, full_ranking_scores = _get_full_ranking(proto_labels, proto_embeds, query_embeds)

    cmc = _measure_cmc(full_ranking_labels, query_labels)

    # rank1_acc = cmc[0]
    # rank3_acc = cmc[2]
    # rank5_acc = cmc[4]
    # rank10_acc = cmc[9]

    mrr = _measure_mrr(full_ranking_labels, query_labels)

    eval_time = time.time() - start_time

    return cmc, mrr, eval_time


def test_retrieval(
    file_name: str,
    image_encoder: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    query_vis_count: int = 32,
    result_vis_count: int = 20
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Evaluate retrieval quality using embedding similarities

    Parameters:
        file_name (str): Output image path for retrieval collage
        image_encoder (torch.nn.Module): Encoder model
        dataloader (DataLoader): Testing dataloader
        device (torch.device): Computation device
        query_vis_count (int): Number of query images visualized
        result_vis_count (int): Number of retrieval results per query.

    Returns:
        tuple:
            auc (float): Global ROC AUC.
            mean_auc (float): Mean per-query AUC.
            fpr (np.ndarray): False positive rates.
            tpr (np.ndarray): True positive rates.
            thr (np.ndarray): Thresholds.
            mean_ap (float): Mean average precision.
    """

    t_start = time.time()

    all_embeddings = []
    all_labels = []
    all_images = []

    with torch.no_grad():

        for images_1, images_2, labels in dataloader:
            # retrieval uses only the first image branch.
            images_1 = images_1.to(device)
            embedding = image_encoder(images_1)

            all_embeddings.append(embedding.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_images.append(images_1.permute(0, 2, 3, 1).cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_images = np.concatenate(all_images, axis=0) * 255

    # similarity matrix between all test embeddings
    similarities = np.dot(all_embeddings, all_embeddings.T)

    logging.info(
        f"SIMILARITIES {similarities.min()} {similarities.max()}, {similarities.shape}"
    )

    collage_ids = np.linspace(
        0,
        all_images.shape[0] - 1,
        query_vis_count
    ).astype(np.int64)

    result_collage = []

    scores = []
    labels = []
    ap = []
    auc = []

    for query_index in range(similarities.shape[0]):

        query_similarity = similarities[query_index]
        query_labels = all_labels[query_index] == all_labels

        # remove self-match from evaluation.
        query_labels[query_index] = False
        query_similarity[query_index] = -1e20

        if np.any(query_labels):
            auc.append(roc_auc_score(query_labels, query_similarity))
            ap.append(average_precision_score(query_labels, query_similarity))

        scores.append(query_similarity)
        labels.append(query_labels)

        if query_index in collage_ids:
            # blue border marks the query image

            query_image = cv2.copyMakeBorder(
                all_images[query_index],
                2, 2, 2, 2,
                cv2.BORDER_CONSTANT,
                value=[255, 0, 0]
            )
            query_image = cv2.copyMakeBorder(
                query_image,
                1, 1, 1, 1,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )

            row = [query_image]
            query_label = all_labels[query_index]

            best_ids = np.argsort(query_similarity)[::-1][:result_vis_count]

            for result_index in best_ids:
                result_image = all_images[result_index]

                # green border means correct retrieval, red means wrong retrieval.
                if all_labels[result_index] == query_label:
                    result_image = cv2.copyMakeBorder(
                        result_image,
                        2, 2, 2, 2,
                        cv2.BORDER_CONSTANT,
                        value=[0, 255, 0]
                    )
                else:
                    result_image = cv2.copyMakeBorder(
                        result_image,
                        2, 2, 2, 2,
                        cv2.BORDER_CONSTANT,
                        value=[0, 0, 255]
                    )

                result_image = cv2.copyMakeBorder(
                    result_image,
                    1, 1, 1, 1,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )
                row.append(result_image)

            result_collage.append(np.concatenate(row, axis=1))

    if result_collage:
        result_collage = np.concatenate(result_collage, axis=0)
        cv2.imwrite(file_name, result_collage)

    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(np.asarray(labels), axis=0)

    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    mean_ap = np.mean(ap)
    mean_auc = np.mean(auc)
    auc = roc_auc_score(labels, scores)

    logger.info(f"TEST TIME: {time.time() - t_start}s")

    return auc, mean_auc, fpr, tpr, thr, mean_ap
