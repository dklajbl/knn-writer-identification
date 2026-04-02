import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass


@dataclass
class OpenSetIdentifMetrics:

    # Thresholds. Shape: (N_thresholds).
    thrs: np.ndarray

    # Shape: (N_thresholds).
    DIR_t: np.ndarray

    # Shape: (N_thresholds).
    FRR_t: np.ndarray

    # Shape: (N_thresholds).
    TAR_t: np.ndarray

    # Shape: (N_thresholds).
    FAR_t: np.ndarray


@dataclass
class ClosedSetIdentifMetrics:

    # Cummulative Match Characteristics.
    # Accuracy for each possible rank [1, N_labels].
    # Accuracy for rank K =
    #   (Num. of queries with correct label within top K predicted labels) / (Total num. of queries).
    # Shape: (N_labels).
    CMC: np.ndarray

    # Mean Reciprocal Rank.
    # Average of (1 / rank of correct label) for each query.
    MRR: np.float32

    # Rank 1 accuracy.
    rank_1_acc: np.float32

    # Rank 3 accuracy.
    rank_3_acc: np.float32

    # Rank 5 accuracy.
    rank_5_acc: np.float32

    # Rank 10 accuracy.
    rank_10_acc: np.float32


@dataclass
class IdentifMetrics:
    closed_set_metrics: ClosedSetIdentifMetrics

    open_set_metrics: OpenSetIdentifMetrics

    eval_time: np.float32


# ==========================================
# CLOSED SET IDENTIFICATION MODEL EVALUATION

def _calc_rank_k_acc(full_ranking_labels: np.ndarray, query_labels: np.ndarray, k=1):
    correct = np.any(
        full_ranking_labels[:, :k] == query_labels[:, None], axis=1)
    return np.mean(correct)


def _calc_cmc(full_ranking_labels: np.ndarray, query_labels: np.ndarray):
    # Calculate CMC (Cumulative Match Characteristic)

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


def _calc_mrr(full_ranking_labels: np.ndarray, query_labels: np.ndarray):
    # Calculate MRR (Mean Reciprocal Rank)

    # create boolean matrix, on each row set a cell to True,
    # if the ranked label for the query matches the query label
    # (there should be only one match for each query)
    # (N_queries, N_labels) = (N_queries, N_labels) == (N_queries, 1)
    matches = (full_ranking_labels == query_labels[:, None])

    # find index of (first and only) label match
    # (N_queries)
    first_correct_idx = np.argmax(matches, axis=1)

    # calculate mean of (1 / (first correct label index + 1))
    mrr = np.mean(1 / (first_correct_idx + 1))

    return mrr


def _measure_closed_set_performance(full_ranking_labels: np.ndarray, query_labels: np.ndarray):
    cmc = _calc_cmc(full_ranking_labels, query_labels)
    mrr = _calc_mrr(full_ranking_labels, query_labels)

    rank_1_acc = cmc[min(0, len(cmc)-1)]
    rank_3_acc = cmc[min(2, len(cmc)-1)]
    rank_5_acc = cmc[min(4, len(cmc)-1)]
    rank_10_acc = cmc[min(9, len(cmc)-1)]

    return ClosedSetIdentifMetrics(CMC=cmc, MRR=mrr,
                                   rank_1_acc=rank_1_acc, rank_3_acc=rank_3_acc,
                                   rank_5_acc=rank_5_acc, rank_10_acc=rank_10_acc)


def _get_accepted_rejected_counts_per_thr(query_scores: np.ndarray, sorted_thrs: np.ndarray):

    N_queries = len(query_scores)

    # for each query score compute the index of the first threshold
    #   that is strictly larger than the query score
    #   == number of thresholds smaller or equal to the query score
    # if there is no such threshold, than the index == N_thresholds
    # (N_queries)
    thr_idxs = np.searchsorted(sorted_thrs, scores)

    print(thr_idxs)

    # for each threshold compute for how many query scores
    #   is the threshold the first strictly larger threshold
    # the last index (= N_thresholds) is the count of queries
    #   with score larger or equal to the largest threshold
    # (N_thresholds+1)
    counts = np.bincount(thr_idxs, minlength=len(sorted_thrs) + 1)[:-1]

    print(counts)

    # compute number of accepted queries for each threshold
    # query is accepted if its score is larger or equal to given threshold
    rejected_counts = counts.cumsum()
    accepted_counts = N_queries - rejected_counts

    print(accepted_counts)
    print(rejected_counts)

    return accepted_counts, rejected_counts

# ========================================
# OPEN SET IDENTIFICATION MODEL EVALUATION


def _calc_open_set_metrics(full_ranking_labels_k: np.ndarray,
                           full_ranking_scores_k: np.ndarray,
                           query_labels: np.ndarray,
                           unknown_query_mask: np.ndarray):

    # compute query counts
    N_queries = len(query_labels)
    N_queries_unknown = np.sum(unknown_query_mask)
    N_queries_known = N_queries - N_queries_unknown

    # (N_queries)
    known_query_mask = ~unknown_query_mask

    # Get best labels and their scores

    # (N_queries)
    top1_labels = full_ranking_labels_k[:, 0]
    top1_scores = full_ranking_scores_k[:, 0]

    # Get statistics for each possible threshold

    # generate thresholds
    # (N_thresholds)
    unique_scores = np.unique(top1_scores)
    sorted_thrs = np.union1d(unique_scores, [0.0, 1.0])

    unknown_query_scores = top1_scores[unknown_query_mask]

    # (N_thresholds)
    unknown_accepted_counts, _ = _get_accepted_rejected_counts_per_thr(
        unknown_query_scores, sorted_thrs)

    known_query_scores = top1_scores[known_query_mask]

    # (N_thresholds)
    _, known_rejected_counts = _get_accepted_rejected_counts_per_thr(
        known_query_scores, sorted_thrs)

    correct_query_mask = (top1_labels == query_labels)
    known_correct_query_mask = correct_query_mask & known_query_mask

    known_correct_query_scores = top1_scores[known_correct_query_mask]

    # (N_thresholds)
    known_accepted_correct_counts, _ = _get_accepted_rejected_counts_per_thr(
        known_correct_query_scores, sorted_thrs)

    # Calcualte metrics

    # DIR = Detection and Identification Rate
    # (num. of KNOWN probes ACCEPTED and CORRECTLY IDENTIFIED) / (total num. of KNOWN probes)
    DIR_t = known_accepted_correct_counts / N_queries_known

    # FRR = False Reject Rate (FNR = False Negative Rate)
    # (num. of KNOWN probes REJECTED) / (total num. of KNOWN probes)
    FRR_t = known_rejected_counts / N_queries_known

    # TAR = True Accept Rate (Acceptance Rate)
    # (num. of KNOWN probes ACCEPTED) / (total num. of KNOWN probes)
    TAR_t = 1 - FRR_t

    # FAR = False Accept Rate (TPR = False Positive Rate)
    # (num. of UNKNOWN probes ACCEPTED) / (total num. of UNKNOWN probes)
    FAR_t = unknown_accepted_counts / N_queries_unknown

    return OpenSetIdentifMetrics(thrs=sorted_thrs,
                                 DIR_t=DIR_t, FRR_t=FRR_t,
                                 TAR_t=TAR_t, FAR_t=FAR_t)


def _measure_open_set_performance(full_ranking_labels: np.ndarray,
                                  full_ranking_scores: np.ndarray,
                                  query_labels: np.ndarray):
    """
    Measure open set performance.

    Parameters:
        full_ranking_labels (np.ndarray): Shape (N_queries, N_labels)
        full_ranking_scores (np.ndarray): Shape (N_queries, N_labels)
        query_labels (np.ndarray): Sorted labels. Shape (N_labels)

    Returns:
        thresholds (np.ndarray): Thresholds. Shape (N_thresholds).
        DIR(t) (np.ndarray): Detection and Identification Rate. Shape (N_thresholds).
        FRR(t) (np.ndarray): False Reject Rate (False Negative Rate). Shape (N_thresholds).
        TAR(t) (np.ndarray): True Accept Rate. Shape (N_thresholds).
        FAR(t) (np.ndarray): False Accept Rate (True Positive Rate). Shape (N_thresholds).
    """

    # get unique lables and the start idexes to same label segments (query_labels is sorted)
    labels = np.unique(query_labels)

    rng = np.random.default_rng(42)

    # TODO repeat following experiment 5 - 10 times

    # STEP 1: Select 20% of labels to withold from gallery

    N_labels = len(labels)
    N_labels_unknown = int(0.2 * len(labels))
    # N_labels_known = N_labels - N_labels_unknown

    # (N_labels_unknown)
    unknown_label_idx = rng.choice(
        N_labels, size=N_labels_unknown, replace=False)

    # (N_labels_unknown)
    unknown_labels = labels[unknown_label_idx]

    # STEP 2: Mark queries as unknown, based on withold labels

    # (N_queries)
    unknown_query_mask = np.isin(query_labels, unknown_labels)
    # known_query_mask = ~unknown_query_mask

    N_queries = len(query_labels)
    # N_queries_unknown = np.sum(unknown_query_mask)
    # N_queries_known = N_queries - N_queries_unknown

    # STEP 3: Remove unknown labels and their scores from ranking

    # True if label is in unknowns
    # (N_queries, N_labels)
    mask = ~np.isin(full_ranking_labels, unknown_labels)

    # (N_queries, N_labels) = full_ranking_labels
    # (N_queries * (N_labels - N_labels_unknown)) = full_ranking_labels[mask]  (boolean indexing flattens array)
    # (N_queries, N_labels - N_labels_unknown) = full_ranking_labels[mask].reshape(N_queries, -1)
    full_ranking_labels_k = full_ranking_labels[mask].reshape(N_queries, -1)
    full_ranking_scores_k = full_ranking_scores[mask].reshape(N_queries, -1)

    return _calc_open_set_metrics(full_ranking_labels_k, full_ranking_scores_k, query_labels, unknown_query_mask)

# ===================================================
# CLOSED AND OPEN SET IDENTIFICATION MODEL EVALUATION


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


def test_identification(
        encoder: torch.nn.Module,
        gallery_dataloader: DataLoader,
        query_dataloader: DataLoader,
        device: torch.device
):
    start_time = time.time()

    # load all gallery samples
    gallery_images, gallery_labels, gallery_embeds = _get_all_from_dataloader(
        encoder, gallery_dataloader, device)

    # load all query samples
    query_images, query_labels, query_embeds = _get_all_from_dataloader(
        encoder, query_dataloader, device)

    # normalize embeddings to lie on unit hypersphere (in case they were not normalized)
    gallery_embeds = gallery_embeds / \
        np.linalg.norm(gallery_embeds, axis=1, keepdims=True)
    query_embeds = query_embeds / \
        np.linalg.norm(query_embeds, axis=1, keepdims=True)

    # get prototypes
    # (N_labels, H, W, C), (N_labels), (N_labels, emb_size)
    proto_images, proto_labels, proto_embeds = _get_prototypes(
        gallery_images, gallery_labels, gallery_embeds)

    # get ranked gallery labels (authors) for each query
    # (based on cosine similarity between embeddings)
    # (N_queries, N_labels), (N_queries, N_labels)
    full_ranking_labels, full_ranking_scores = _get_full_ranking(
        proto_labels, proto_embeds, query_embeds)

    closed_set_metrics = _measure_closed_set_performance(
        full_ranking_labels, query_labels)

    open_set_metrics = _measure_open_set_performance(
        full_ranking_labels, full_ranking_scores, query_labels)

    eval_time = time.time() - start_time

    return IdentifMetrics(closed_set_metrics=closed_set_metrics,
                          open_set_metrics=open_set_metrics,
                          eval_time=eval_time)


if __name__ == "__main__":
    sorted_thrs = np.array([0.2, 0.4, 0.6, 0.8])
    scores = np.array([
        0.05,
        0.21, 0.4,
        0.41, 0.45, 0.55,
        0.62, 0.77, 0.78, 0.79,
        0.81, 0.92, 0.97
    ])

    _get_accepted_rejected_counts_per_thr(scores, sorted_thrs)
