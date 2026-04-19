import numpy as np
from metrics import CSI_Metrics


def _calc_CSI_cmc(full_ranking_labels: np.ndarray,
                  query_labels: np.ndarray):
    """
    Calculate CMC (Cumulative Match Characteristic) = accuracy for each possible rank K in 1...N_labels.
    Accuracy for rank K = (Num. of queries with correct label within top K predicted labels) / (Total num. of queries).

    Parameters:
        full_ranking_labels (np.ndarray, shape=(N_queries, N_labels), dtype=int):
            All ranked predicted labels for each query.

        query_labels (np.ndarray, shape=(N_queries,), dtype=int):
            Query labels (ground truth).

    Returns:
        cmc (bp.ndarray, shape=(N_labels), dtype=np.float32): Cumulative Match Characteristic.
    """

    # create boolean matrix, on each row set a cell to True,
    # if the ranked label for the query matches the query label
    # (there should be only one match for each query)
    # (N_queries, N_labels)
    matches = (full_ranking_labels == query_labels[:, None])

    # check if each query has match in label ranking
    # (N_queries)
    has_match = matches.any(axis=1)

    # find index of (first and only) label match
    # (N_queries)
    first_correct_idx = np.argmax(matches, axis=1)

    # if there was no match replace the index with infinity
    first_correct_idx = np.where(has_match, first_correct_idx, np.inf)

    n_labels = full_ranking_labels.shape[1]
    cmc_curve = np.zeros(n_labels, dtype=np.float32)

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

        query_labels (np.ndarray, shape=(N_queries,), dtype=int):
            Query labels (ground truth).

    Returns:
        mrr (np.float32): Mean Reciprocal Rank.
    """

    # create boolean matrix, on each row set a cell to True,
    # if the ranked label for the query matches the query label
    # (there should be only one match for each query)
    # (N_queries, N_labels) = (N_queries, N_labels) == (N_queries, 1)
    matches = (full_ranking_labels == query_labels[:, None])

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

    rank_1_acc = CSI_Metrics.get_rank_k_acc(cmc, 1)
    rank_5_acc = CSI_Metrics.get_rank_k_acc(cmc, 5)
    rank_10_acc = CSI_Metrics.get_rank_k_acc(cmc, 10)

    mrr = _calc_CSI_mrr(full_ranking_labels, query_labels)

    return CSI_Metrics(cmc=cmc,
                       rank_1_acc=rank_1_acc,
                       rank_5_acc=rank_5_acc,
                       rank_10_acc=rank_10_acc,
                       mrr=mrr)
