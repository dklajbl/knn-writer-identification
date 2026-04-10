import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass


@dataclass(frozen=True)
class OSI_OSCR_Curve:
    # OSCR: TPIR x FPIR
    x_fpir: np.ndarray
    x_fpir_std: np.ndarray | None = None
    y_tpir: np.ndarray
    y_tpir_std: np.ndarray | None = None
    y_thr: np.ndarray
    y_thr_std: np.ndarray | None = None
    auc: np.float32
    auc_std: np.float32 | None = None


@dataclass(frozen=True)
class OSI_ROC_Curve:
    # ROC: TPR x FPR(FPIR)
    x_fpr: np.ndarray
    x_fpr_std: np.ndarray | None = None
    y_tpr: np.ndarray
    y_tpr_std: np.ndarray | None = None
    y_thr: np.ndarray
    y_thr_std: np.ndarray | None = None
    auc: np.float32
    auc_std: np.float32 | None = None


@dataclass(frozen=True)
class OSI_DET_Curve:
    # DET: FNR(FNIR) vs FPR
    x_fpr: np.ndarray
    x_fpr_std: np.ndarray | None = None
    y_fnr: np.ndarray
    y_fnr_std: np.ndarray | None = None
    y_thr: np.ndarray
    y_thr_std: np.ndarray | None = None
    auc: np.float32
    auc_std: np.float32 | None = None


@dataclass(frozen=True)
class OSI_FPIR_OpPoint:
    # TPIR and Threshold at specific FPIR operating points
    fpir: np.float32
    fpir_std: np.float32 | None = None
    tpir: np.float32
    tpir_std: np.float32 | None = None
    thr: np.float32
    thr_std: np.float32 | None = None


@dataclass(frozen=True)
class OSI_EER:
    # EER = Equal Error Rate
    # Point on DET curve, where FPIR ≈ FNIR
    val: np.float32
    val_std: np.float32 | None = None

    # EER_thr = Equal Error Rate Threshold
    # Threshold, where FPIR ≈ FNIR
    thr: np.float32
    thr_std: np.float32 | None = None


@dataclass(frozen=True)
class OSI_Metrics:
    """
    Stores open-set identification metrics.
    """

    oscr_curve: OSI_OSCR_Curve
    roc_curve: OSI_ROC_Curve
    det_curve: OSI_DET_Curve
    main_fpir_op_points: dict[float, OSI_FPIR_OpPoint]
    eer: OSI_EER


@dataclass(frozen=True)
class CSI_Metrics:
    """
    Stores closed-set identification metrics.
    """

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
class IdnetificationMetrics:
    """
    Stores (open-set and closed-set) identification metrics.
    """

    # Closed-set Identification metrics
    csi_metrics: CSI_Metrics

    # Open-set identification metrics
    osi_metrics: OSI_Metrics

    # Measured time of evaluation
    eval_time: np.float32


# ==========================================
# CLOSED SET IDENTIFICATION MODEL EVALUATION

def _calc_CSI_cmc(full_ranking_labels: np.ndarray,
                  query_labels: np.ndarray):
    """
    Calculate CMC (Cumulative Match Characteristic) = accuracy for each possible rank K in 1...N_labels.
    Accuracy for rank K = (Num. of queries with correct label within top K predicted labels) / (Total num. of queries).

    Parameters:
        full_ranking_labels (np.ndarray, shape=(N_queries, N_labels), dtype=int):
            All ranked predicted labels for each query.

        query_labels (np.ndarray, shape=(N_labels), dtype=int):
            Query labels (ground truth).

    Returns:
        cmc (bp.ndarray, shape=(N_labels), dtype=np.float32): Cumulative Match Characteristic.
    """

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


def _calc_CSI_mrr(full_ranking_labels: np.ndarray,
                  query_labels: np.ndarray):
    """
    Calculate MRR (Mean Reciprocal Rank) = average of (1 / (correct label rank)) for each query.

    Parameters:
        full_ranking_labels (np.ndarray, shape=(N_queries, N_labels), dtype=int):
            All ranked predicted labels for each query.

        query_labels (np.ndarray, shape=(N_labels), dtype=int):
            Query labels (ground truth).

    Returns:
        mrr (np.float32): Mean Reciprocal Rank.
    """

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


def _calc_CSI_metrics(full_ranking_labels: np.ndarray,
                      query_labels: np.ndarray):
    """
    Calculate closed-set identification metrics.

    Parameters:
        full_ranking_labels (np.ndarray, shape=(N_queries, N_labels), dtype=int):
            All ranked predicted labels for each query.

        query_labels (np.ndarray, shape=(N_labels), dtype=int):
            Query labels (ground truth).

    Returns:
        closed_set_identif_metrics (OpenSetIdentifMetrics):
            Dataclass containing calcualted closed-set identification metrics.
    """

    cmc = _calc_CSI_cmc(full_ranking_labels, query_labels)
    mrr = _calc_CSI_mrr(full_ranking_labels, query_labels)

    rank_1_acc = cmc[min(0, len(cmc)-1)]
    rank_3_acc = cmc[min(2, len(cmc)-1)]
    rank_5_acc = cmc[min(4, len(cmc)-1)]
    rank_10_acc = cmc[min(9, len(cmc)-1)]

    return CSI_Metrics(CMC=cmc, MRR=mrr,
                       rank_1_acc=rank_1_acc, rank_3_acc=rank_3_acc,
                       rank_5_acc=rank_5_acc, rank_10_acc=rank_10_acc)

# ========================================
# OPEN SET IDENTIFICATION MODEL EVALUATION


def _calc_OSI_metrics_from_counts(
        thrs_sorted: np.ndarray,
        N_queries_known: int,
        N_queries_unknown: int,
        unknown_accepted_counts: np.ndarray,
        unknown_rejected_counts: np.ndarray,
        known_accepted_correct_counts: np.ndarray,
        known_accepted_wrong_counts: np.ndarray,
        known_rejected_counts: np.ndarray):

    # TPIR = True Positive Identification Rate.
    #   (== DIR = Detection and Identification Rate)
    # TPIR = (num. of KNOWN queries ACCEPTED and CORRECTLY IDENTIFIED)
    #           / (total num. of KNOWN queries).
    # Shape: (N_thresholds).
    TPIR_t = known_accepted_correct_counts / N_queries_known

    # FNIR = False Negative Identification Rate.
    # FNIR = (num. of KNOWN queries REJECTED or (ACCEPTED and INCORRECTLY IDENTIFIED))
    #           / (total num. of KNOWN queries)
    FNIR_t = 1 - TPIR_t

    # FPIR = False Positive Identification Rate per threshold.
    #   (== FAR = False Accept Rate)
    # FPIR = (num. of UNKNOWN queries ACCEPTED) / (total num. of UNKNOWN queries)
    # Shape: (N_thresholds).
    FPIR_t = unknown_accepted_counts / N_queries_unknown

    # FPR = False Positive Rate = FPIR
    FPR_t = FPIR_t

    # FNR = False Negative Rate (Known Rejection Rate)
    # FNR = (num. of KNOWN queries REJECTED) / (total num. of KNOWN queries)
    FNR_t = known_rejected_counts / N_queries_known

    # TPR = True Positive Rate (Known Acceptance Rate)
    # TPR = (num. of KNOWN queries ACCEPTED) / (total num. of KNOWN queries)
    TPR_t = 1 - FNR_t

    # OSCR: TPIR x FPIR

    # have to guarantee, that X value for inteprolation are stricly monotonously ascending
    _, fpir_unique_idx = np.unique(FPIR_t, return_index=True)
    fpir_order = fpir_unique_idx[np.argsort(FPIR_t[fpir_unique_idx])]

    fpir_t_sorted = FPIR_t[fpir_order]
    tpir_t_sorted_by_fpir = TPIR_t[fpir_order]
    thrs_sorted_by_fpir = thrs_sorted[fpir_order]

    oscr_x_fpir = np.logspace(-4, 0, num=200)  # 0.0001=1e-4 -> 1
    oscr_y_tpir = np.interp(
        oscr_x_fpir, fpir_t_sorted, tpir_t_sorted_by_fpir)
    oscr_y_thr = np.interp(
        oscr_x_fpir, fpir_t_sorted, thrs_sorted_by_fpir)
    oscr_auc = np.trapz(tpir_t_sorted_by_fpir, fpir_t_sorted)

    oscr_curve = OSI_OSCR_Curve(x_fpir=oscr_x_fpir,
                                y_tpir=oscr_y_tpir,
                                y_thr=oscr_y_thr,
                                auc=oscr_auc)

    # TPIR @ FPIR=10%, TPIR @ FPIR=1%, TPIR @ FPIR=0.1%, TPIR @ FPIR=0.01%
    fpir_targets = np.array([1e-1, 1e-2, 1e-3, 1e-4])
    main_fpir_op_points = {
        fpir: {
            OSI_FPIR_OpPoint(
                fpir=fpir,
                tpir=np.interp(fpir, fpir_t_sorted, tpir_t_sorted_by_fpir),
                thr=np.interp(fpir, fpir_t_sorted, thrs_sorted_by_fpir)
            )
        }
        for fpir in fpir_targets
    }

    # ROC: TPR x FPR(FPIR)
    tpr_sorted_by_fpir = TPR_t[fpir_order]

    roc_x_fpr = np.logspace(-4, 0, num=200)
    roc_y_tpr = np.interp(roc_x_fpr, fpir_t_sorted, tpr_sorted_by_fpir)
    roc_y_thr = np.interp(roc_x_fpr, fpir_t_sorted, thrs_sorted_by_fpir)
    roc_auc = np.trapz(tpr_sorted_by_fpir, fpir_t_sorted)

    roc_curve = OSI_ROC_Curve(x_fpr=roc_x_fpr,
                              y_tpr=roc_y_tpr,
                              y_thr=roc_y_thr,
                              auc=roc_auc)

    # DET: FNR(FNIR) vs FPR
    _, fpr_unique_idx = np.unique(FPR_t, return_index=True)
    fpr_order = fpr_unique_idx[np.argsort(FPR_t[fpr_unique_idx])]

    fpr_t_sorted = FPR_t[fpr_order]
    fnr_t_sorted_by_fpr = FNR_t[fpr_order]
    thrs_sorted_by_fpr = thrs_sorted[fpr_order]

    det_x_fpr = np.logspace(-4, 0, num=200)
    det_y_fnr = np.interp(det_x_fpr, fpr_t_sorted, fnr_t_sorted_by_fpr)
    det_y_thr = np.interp(det_x_fpr, fpr_t_sorted, thrs_sorted_by_fpr)
    det_auc = np.trapz(fnr_t_sorted_by_fpr, fpr_t_sorted)

    det_curve = OSI_DET_Curve(x_fpr=det_x_fpr,
                              y_fnr=det_y_fnr,
                              y_thr=det_y_thr,
                              auc=det_auc)

    # Equal Error Rate: FPIR ≈ FNIR
    eer_idx = np.argmin(np.abs(FPIR_t - FNIR_t))
    eer_val = (FPIR_t[eer_idx] + FNIR_t[eer_idx]) / 2
    eer_thr = thrs_sorted[eer_idx]

    eer = OSI_EER(val=eer_val, thr=eer_thr)

    return OSI_Metrics(oscr_curve=oscr_curve,
                       roc_curve=roc_curve,
                       det_curve=det_curve,
                       main_fpir_op_points=main_fpir_op_points,
                       eer=eer)


def _get_OSI_acc_rej_counts_per_thr(query_scores: np.ndarray,
                                    sorted_thrs: np.ndarray):
    """
    Calculate counts of accepted and rejected queries for each threshold.
    Query is accepted if its score is strictly larger than the threshold.

    Parameters:
        query_scores (np.ndarray, shape=(N_queries, N_labels), dtype=np.float32):
            Score for each query.

        sorted_thrs (np.ndarray, shape=(N_thresholds), dtype=np.float32):
            Sorted thresholds in ascending order.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            accepted_counts (np.ndarray, shape=(N_thresholds), dtype=np.int):\
                Counts of accepted queries for each threshold.

            rejected_counts (np.ndarray, shape=(N_thresholds), dtype=np.int):\
                Counts of rejected queries for each threshold.
    """

    N_queries = len(query_scores)

    # for each query score compute the index of the first threshold
    #   that is strictly larger than the query score
    #   == number of thresholds smaller or equal to the query score
    # if there is no such threshold, than the index == N_thresholds
    # (N_queries)
    thr_idxs = np.searchsorted(sorted_thrs, scores)

    # for each threshold compute for how many query scores
    #   is the threshold the first strictly larger threshold
    # the last index (= N_thresholds) is the count of queries
    #   with score larger or equal to the largest threshold
    # (N_thresholds+1)
    counts = np.bincount(thr_idxs, minlength=len(sorted_thrs) + 1)[:-1]

    # compute number of accepted queries for each threshold
    # query is accepted if its score is larger or equal to given threshold
    rejected_counts = counts.cumsum()
    accepted_counts = N_queries - rejected_counts

    return accepted_counts, rejected_counts


def _calc_OSI_metrics_masked(full_ranking_labels_k: np.ndarray,
                             full_ranking_scores_k: np.ndarray,
                             query_labels: np.ndarray,
                             unknown_query_mask: np.ndarray):
    """
    Calculate open-set identification metrics with masked unknown queries
        (query label does not exist in gallery).
    Helper function called by `_calc_open_set_identif_metrics`.

    Parameters:
        full_ranking_labels_k (np.ndarray, shape=(N_queries, N_labels), dtype=int):
            All ranked predicted labels for each query excluding unknown labels.

        full_ranking_scores_k (np.ndarray. shape=(N_queries, N_labels), dtype=np.float32):
            Scores of all ranked predicted labels for each query excluding unknown labels.

        query_labels (np.ndarray, shape=(N_labels), dtype=int): Query labels (ground truth).

        unknown_query_mask (np.ndarray, shape=(N_labels), dtype=bool):
            Boolean mask over queries. True if the query label is unknown.

    Returns:
        open_set_identif_metrics (OpenSetIdentifMetrics):
            Dataclass containing calculated open-set identification metrics.
    """

    # STEP 1: Preparation

    # compute query counts
    N_queries = len(query_labels)
    N_queries_unknown = np.sum(unknown_query_mask)
    N_queries_known = N_queries - N_queries_unknown

    # get best labels and their scores
    # (N_queries)
    top1_labels = full_ranking_labels_k[:, 0]
    top1_scores = full_ranking_scores_k[:, 0]

    # STEP 2: Get thresholds == all uniques scores in sorted order

    # generate thresholds
    # (N_thresholds)
    unique_scores = np.unique(top1_scores)
    sorted_thrs = np.union1d(unique_scores, [-1.0, 1.0])

    # STEP 3: Compute query count statistics (rejected, accepted, ...) for each possible threshold

    # unknown sample count statistics

    unknown_query_scores = top1_scores[unknown_query_mask]

    unknown_accepted_counts, unknown_rejected_counts = _get_OSI_acc_rej_counts_per_thr(
        unknown_query_scores, sorted_thrs)

    # known sample count statistics

    known_query_mask = ~unknown_query_mask
    known_query_scores = top1_scores[known_query_mask]

    known_accepted_counts, known_rejected_counts = _get_OSI_acc_rej_counts_per_thr(
        known_query_scores, sorted_thrs)

    correct_query_mask = (top1_labels == query_labels)
    known_correct_query_mask = correct_query_mask & known_query_mask
    known_correct_query_scores = top1_scores[known_correct_query_mask]

    known_accepted_correct_counts, _ = _get_OSI_acc_rej_counts_per_thr(
        known_correct_query_scores, sorted_thrs)

    known_accepted_wrong_counts = known_accepted_counts - known_accepted_correct_counts

    # STEP 4: Calcualte (per threshold) metrics

    return _calc_OSI_metrics_from_counts(sorted_thrs=sorted_thrs,
                                         N_queries_known=N_queries_known,
                                         N_queries_unknown=N_queries_unknown,
                                         unknown_accepted_counts=unknown_accepted_counts,
                                         unknown_rejected_counts=unknown_rejected_counts,
                                         known_accepted_correct_counts=known_accepted_correct_counts,
                                         known_accepted_wrong_counts=known_accepted_wrong_counts,
                                         known_rejected_counts=known_rejected_counts)


def _calc_OSI_metrics(full_ranking_labels: np.ndarray,
                      full_ranking_scores: np.ndarray,
                      query_labels: np.ndarray):
    """
    Calculate open-set identification metrics.

    Parameters:
        full_ranking_labels (np.ndarray, shape=(N_queries, N_labels), dtype=int):
            All ranked predicted labels for each query.

        full_ranking_scores (np.ndarray, shape=(N_queries, N_labels), dtype=np.float32):
            Scores of all ranked predicted labels for each query.

        query_labels (np.ndarray, shape=(N_labels), dtype=int): Query labels (ground truth).

    Returns:
        open_set_identif_metrics (OpenSetIdentifMetrics):
            Dataclass containing calculated open-set identification metrics.
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

    # boolean mask over queries,
    #   where True indicates that the query label is in unknowns
    # (N_queries)
    unknown_query_mask = np.isin(query_labels, unknown_labels)

    N_queries = len(query_labels)

    # STEP 3: Remove unknown labels and their scores from ranking

    # create boolean mask over full_ranking_labels 2D array,
    #   where True indicates that the label is known
    # (N_queries, N_labels)
    known_mask = ~np.isin(full_ranking_labels, unknown_labels)

    # remove unknown labels and their scores
    # (N_queries, N_labels) = full_ranking_labels
    # (N_queries * (N_labels - N_labels_unknown)) = full_ranking_labels[mask]  (boolean indexing flattens array)
    # (N_queries, N_labels - N_labels_unknown) = full_ranking_labels[mask].reshape(N_queries, -1)
    full_ranking_labels_k = full_ranking_labels[known_mask].reshape(
        N_queries, -1)
    full_ranking_scores_k = full_ranking_scores[known_mask].reshape(
        N_queries, -1)

    return _calc_OSI_metrics_masked(full_ranking_labels_k, full_ranking_scores_k,
                                    query_labels, unknown_query_mask)

# ===================================================
# CLOSED AND OPEN SET IDENTIFICATION MODEL EVALUATION


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
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            all_images (np.ndarray, shape=(N_samples, H, W, C), dtype=np.float32): Array of loaded images.

            all_labels (np.ndarray, shape=(N_samples,), dtype=int): Array of labels.

            all_embeds (np.ndarray, shape=(N_samples, Embed_size), dtype=np.float32): Array of embeddings.
    """

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

            # (B, Embed_size)
            embeds_np = embeds.cpu().numpy().astype(np.float32)

            all_images.append(images_np)
            all_labels.append(labels_np)
            all_embeds.append(embeds_np)

    # (N_samples, H, W, C)
    all_images = np.concatenate(all_images, axis=0)
    # (N_samples)
    all_labels = np.concatenate(all_labels, axis=0)
    # (N_samples, Embed_size)
    all_embeds = np.concatenate(all_embeds, axis=0)

    # sort by labels
    perm = np.argsort(all_labels)
    all_images = all_images[perm]
    all_labels = all_labels[perm]
    all_embeds = all_embeds[perm]

    return all_images, all_labels, all_embeds


def _get_prototypes(gallery_images: np.ndarray,
                    gallery_labels: np.ndarray,
                    gallery_embeds: np.ndarray):
    """
    Create prototypes from gallery samples.
    Prototype embedding for specific label is computed as average of gallery embeddings with the same label.

    Parameters:
        gallery_images (np.ndarray, shape=(N_samples, H, W, C), dtype=np.float32):
            Array of gallery image samples.

        gallery_labels (np.ndarray, shape=(N_samples), dtype=int):
            Array of gallery labels.

        gallery_embeds (np.ndarray, shape=(N_samples, Embed_size), dtype=np.float32):
            Array of gallery embeddings.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            proto_images (np.ndarray, shape=(N_labels, H, W, C), dtype=np.float32):\
                Array of prototype sample images (one arbitrary image from gallery).

            proto_labels (np.ndarray, shape=(N_labels,), dtype=int): Array of prototype labels.

            proto_embeds (np.ndarray, shape=(N_labels, Embed_size), dtype=np.float32): Array of prototype embeddings.
    """

    # gallery labels are already sorted (and also are the corresponding embeddings)
    # so np.unique will return start indices of the same label segments
    # (N_labels), (N_labels)
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

    # (N_labels, Embed_size)
    proto_embeds = np.stack(proto_embeds)

    # (N_labels, H, W, C)
    proto_images = np.stack(proto_images)

    # normalize prototype embeddings
    proto_embeds = proto_embeds / \
        np.linalg.norm(proto_embeds, axis=1, keepdims=True)

    # (N_labels, H, W, C), (N_labels), (N_labels, Embed_size)
    return proto_images, proto_labels, proto_embeds


def _get_full_ranking(proto_labels: np.ndarray,
                      proto_embeds: np.ndarray,
                      query_embeds: np.ndarray):

    full_ranking_idxs = []
    full_ranking_scores = []

    # iterate over batches of queries
    query_batch_size = 1024
    for i in range(0, len(query_embeds), query_batch_size):

        # (query_batch_size, emb_size)
        query_batch = query_embeds[i:i+query_batch_size]

        # computing cosine similarity (prototype and query embeddings are already normalized)
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

    # map indices to actual labels
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

    csi_metrics = _calc_CSI_metrics(
        full_ranking_labels, query_labels)

    osi_metrics = _calc_OSI_metrics(
        full_ranking_labels, full_ranking_scores, query_labels)

    eval_time = time.time() - start_time

    return IdnetificationMetrics(csi_metrics=csi_metrics,
                                 osi_metrics=osi_metrics,
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

    _get_OSI_acc_rej_counts_per_thr(scores, sorted_thrs)
