import numpy as np
from metrics import CSI_Metrics


def _calc_CSI_cmc(class_labels_ranked: np.ndarray,
                  query_labels: np.ndarray):
    """
    Calculate CMC (Cumulative Match Characteristic) = accuracy for each possible rank K in 1...N_labels.
    Accuracy for rank K = (Num. of queries with correct label within top K predicted labels) / (Total num. of queries).

    Parameters:
        class_labels_ranked (np.ndarray, shape=(N_queries, N_labels), dtype=int):
            Ranked class labels for each query.

        query_labels (np.ndarray, shape=(N_queries,), dtype=int):
            Query labels (ground truth).

    Returns:
        cmc (bp.ndarray, shape=(N_labels), dtype=np.float32): Cumulative Match Characteristic.
    """

    # create boolean matrix, on each row set a cell to True,
    # if the ranked label for the query matches the query label
    # (there should be only one match for each query)
    # (N_queries, N_labels)
    matches = (class_labels_ranked == query_labels[:, None])

    # check if each query has match in label ranking
    # (N_queries)
    has_match = matches.any(axis=1)

    # find index of (first and only) label match
    # (N_queries)
    first_correct_idx = np.argmax(matches, axis=1)

    # if there was no match replace the index with infinity
    first_correct_idx = np.where(has_match, first_correct_idx, np.inf)

    n_labels = class_labels_ranked.shape[1]
    cmc_curve = np.zeros(n_labels, dtype=np.float32)

    for k in range(n_labels):
        # check if first correct label was within rank k for each query
        # then calculate mean of this boolean array
        cmc_curve[k] = np.mean(first_correct_idx <= k)

    return cmc_curve


def _calc_CSI_mrr(class_labels_ranked: np.ndarray,
                  query_labels: np.ndarray):
    """
    Calculate MRR (Mean Reciprocal Rank) = average of (1 / (correct label rank)) for each query.

    Parameters:
        class_labels_ranked (np.ndarray, shape=(N_queries, N_labels), dtype=int):
            Ranked class labels for each query.

        query_labels (np.ndarray, shape=(N_queries,), dtype=int):
            Query labels (ground truth).

    Returns:
        mrr (np.float32): Mean Reciprocal Rank.
    """

    # create boolean matrix, on each row set a cell to True,
    # if the ranked label for the query matches the query label
    # (there should be only one match for each query)
    # (N_queries, N_labels) = (N_queries, N_labels) == (N_queries, 1)
    matches = (class_labels_ranked == query_labels[:, None])

    # check if each query has match in label ranking
    # (N_queries)
    has_match = matches.any(axis=1)

    # find index of (first and only) label match
    # (N_queries)
    first_correct_idx = np.argmax(matches, axis=1)

    # calculate mean of (1 / (first correct label index + 1))
    rr = np.zeros(matches.shape[0], dtype=np.float32)
    rr[has_match] = 1.0 / (first_correct_idx[has_match] + 1)
    mrr = rr.mean()

    return mrr


def _calc_mAP(gallery_labels_ranked: np.ndarray, query_labels: np.ndarray):
    # mask over gallery_labels_ranked
    #   1.0 -> the label matches the label of the query
    #   0.0 -> the label does not match the label of the query
    # (N_queries, N_gallery)
    relevant_mask = (gallery_labels_ranked == query_labels[:, None]).astype(np.float32)

    # counts number relevant items for each rank
    # (N_queries, N_gallery)
    cumsum_relevant = np.cumsum(relevant_mask, axis=1)

    # count total number of relevant items for each query
    # (N_queries)
    num_relevant = relevant_mask.sum(axis=1)

    # create sequence of ranks (1, 2, ..., N_gallery)
    # (N_gallery)
    ranks = np.arange(1, relevant_mask.shape[1] + 1)

    # calculate precision for each rank
    # rank k precision = (num. relevant items in top k) / k
    # (N_queries, N_gallery)
    precisions = cumsum_relevant / ranks

    # calculate average precision for each query
    # query precision
    #   = (1 / num. relevant items for query)
    #       * (sum over precisions at relevant positions of the query)
    # (N_queries)
    average_precisions = np.where(
        num_relevant > 0,
        (precisions * relevant_mask).sum(axis=1) / num_relevant,
        0.0
    )

    # calcualte mean average precision
    # == average of average precisions of each query
    # scalar
    mAP = average_precisions.mean()

    return mAP


def _calc_CSI_metrics(gallery_labels_ranked: np.ndarray,
                      class_labels_ranked: np.ndarray,
                      query_labels: np.ndarray):
    """
    Calculate closed-set identification metrics.

    Parameters:
        gallery_labels_ranked (np.ndarray, shape=(N_queries, N_labels), dtype=int):
            Ranked gallery sample labels for each query.

        class_labels_ranked (np.ndarray, shape=(N_queries, N_labels), dtype=int):
            Ranked class labels for each query.

        query_labels (np.ndarray, shape=(N_labels), dtype=int):
            Query labels (ground truth).

    Returns:
        closed_set_identif_metrics (OpenSetIdentifMetrics):
            Dataclass containing calcualted closed-set identification metrics.
    """

    mAP = _calc_mAP(gallery_labels_ranked, query_labels)

    cmc = _calc_CSI_cmc(class_labels_ranked, query_labels)

    rank_1_acc = CSI_Metrics.get_rank_k_acc(cmc, 1)
    rank_5_acc = CSI_Metrics.get_rank_k_acc(cmc, 5)
    rank_10_acc = CSI_Metrics.get_rank_k_acc(cmc, 10)

    mrr = _calc_CSI_mrr(class_labels_ranked, query_labels)

    return CSI_Metrics(mAP=mAP,
                       cmc=cmc,
                       rank_1_acc=rank_1_acc,
                       rank_5_acc=rank_5_acc,
                       rank_10_acc=rank_10_acc,
                       mrr=mrr)
