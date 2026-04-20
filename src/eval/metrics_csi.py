import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class CSI_Metrics:
    """
    Stores closed-set identification metrics.
    """

    # Mean Average Precision
    # mAP = (1 / num. queries) * sum over AP_q for each query
    # AP_q = average percision of query q =
    #   = (1 / num. relevant items for query)
    #       * (sum over precision@k for each rank)
    # precision@k = (num. relevant items in top k) / k
    mAP: np.float32

    # Cummulative Match Characteristics.
    # Accuracy for each possible rank [1, N_labels].
    # Accuracy for rank K =
    #   (Num. of queries with correct label within top K predicted labels) / (Total num. of queries).
    # Shape: (N_labels).
    cmc: np.ndarray

    # Rank 1 accuracy = cmc[0]
    rank_1_acc: np.float32

    # Rank 5 accuracy = cmc[4]
    rank_5_acc: np.float32

    # Rank 10 accuracy = cmc[9]
    rank_10_acc: np.float32

    # Mean Reciprocal Rank.
    # Average of (1 / rank of correct label) for each query.
    mrr: np.float32

    @staticmethod
    def get_rank_k_acc(cmc: np.ndarray, k: int) -> np.float32:
        """ Get rank k accuracy. """
        if k < 1:
            raise ValueError(f"Rank k must be >= 1, got {k}")

        if len(cmc) == 0:
            raise ValueError("CMC is empty")

        if k > len(cmc):
            raise ValueError(f"Rank k={k} exceeds max rank={len(cmc)}")

        rank_k_acc = cmc[k - 1]

        return rank_k_acc
