import numpy as np
from dataclasses import dataclass
from metrics import OSI_Metrics, OSI_OSCR_Curve, OSI_ROC_Curve, OSI_DET_Curve, OSI_EER, OSI_FPIR_OpPoint


@dataclass(frozen=True)
class OSI_ThresholdDecisionStats:
    """ Per-threshold decision statistics for Open-Set Identification (OSI). """

    thrs_sorted: np.ndarray

    N_queries_known: int
    N_queries_unknown: int
    unknown_accepted_counts: np.ndarray
    unknown_rejected_counts: np.ndarray
    known_accepted_correct_counts: np.ndarray
    known_accepted_wrong_counts: np.ndarray
    known_rejected_counts: np.ndarray

    def __eq__(self, other):
        if not isinstance(other, OSI_ThresholdDecisionStats):
            return False

        return (
            np.array_equal(self.thrs_sorted, other.thrs_sorted) and
            self.N_queries_known == other.N_queries_known and
            self.N_queries_unknown == other.N_queries_unknown and
            np.array_equal(self.unknown_accepted_counts, other.unknown_accepted_counts) and
            np.array_equal(self.unknown_rejected_counts, other.unknown_rejected_counts) and
            np.array_equal(self.known_accepted_correct_counts, other.known_accepted_correct_counts) and
            np.array_equal(self.known_accepted_wrong_counts, other.known_accepted_wrong_counts) and
            np.array_equal(self.known_rejected_counts,
                           other.known_rejected_counts)
        )


def _is_asc(X):
    return np.all(np.diff(X) >= 0)


def _is_desc(X):
    return np.all(np.diff(X) <= 0)


def _calc_OSI_metrics_from_thr_decision_stats(thr_stats: OSI_ThresholdDecisionStats):

    assert _is_asc(thr_stats.thrs_sorted)
    assert _is_desc(thr_stats.known_accepted_correct_counts)
    assert _is_desc(thr_stats.known_accepted_wrong_counts)
    assert _is_asc(thr_stats.known_rejected_counts)
    assert _is_desc(thr_stats.unknown_accepted_counts)
    assert _is_asc(thr_stats.unknown_rejected_counts)

    # TPIR = True Positive Identification Rate.
    #   (== DIR = Detection and Identification Rate)
    # TPIR = (num. of KNOWN queries ACCEPTED and CORRECTLY IDENTIFIED)
    #           / (total num. of KNOWN queries).
    # Descending. THR increases -> TPIR decreases.
    # Shape: (N_thresholds).
    TPIR_t = thr_stats.known_accepted_correct_counts / thr_stats.N_queries_known

    # FNIR = False Negative Identification Rate.
    # FNIR = (num. of KNOWN queries REJECTED or (ACCEPTED and INCORRECTLY IDENTIFIED))
    #           / (total num. of KNOWN queries)
    # Ascending. THR increases -> FNIR increases.
    # Shape: (N_thresholds).
    FNIR_t = 1 - TPIR_t

    # FPIR = False Positive Identification Rate per threshold.
    #   (== FAR = False Accept Rate)
    # FPIR = (num. of UNKNOWN queries ACCEPTED) / (total num. of UNKNOWN queries)
    # Descending. THR increases -> FPIR decreases.
    # Shape: (N_thresholds).
    FPIR_t = thr_stats.unknown_accepted_counts / thr_stats.N_queries_unknown

    # FPR = False Positive Rate = FPIR
    FPR_t = FPIR_t

    # FNR = False Negative Rate (Known Rejection Rate)
    # FNR = (num. of KNOWN queries REJECTED) / (total num. of KNOWN queries)
    # Ascending. THR increases -> FNR increases.
    FNR_t = thr_stats.known_rejected_counts / thr_stats.N_queries_known

    # TPR = True Positive Rate (Known Acceptance Rate)
    # TPR = (num. of KNOWN queries ACCEPTED) / (total num. of KNOWN queries)
    # Descending. THR increases -> TPR decreases.
    TPR_t = 1 - FNR_t

    # OSCR: x=FPIR vs y=TPIR

    #   Thrs are ascending, FPIR_t is descending, TPIR_t is descending
    #   ---> but OSCR requires -->
    #   FPIR_t is ascending, TPIR_t is ascending -> Thrs must be descending

    fpir_ascending = FPIR_t[::-1]
    tpir_ascending = TPIR_t[::-1]
    thrs_descending = thr_stats.thrs_sorted[::-1]

    oscr_x_fpir = np.linspace(0.0, 1.0, 500)
    oscr_y_tpir = np.interp(
        oscr_x_fpir, fpir_ascending, tpir_ascending,
        left=0, right=tpir_ascending[-1])
    oscr_y_thr = np.interp(
        oscr_x_fpir, fpir_ascending, thrs_descending,
        left=thrs_descending[0], right=thrs_descending[-1])
    oscr_auc = np.trapz(tpir_ascending, fpir_ascending)

    oscr_curve = OSI_OSCR_Curve(x_fpir=oscr_x_fpir, x_fpir_std=np.zeros_like(oscr_x_fpir),
                                y_tpir=oscr_y_tpir, y_tpir_std=np.zeros_like(oscr_y_tpir),
                                y_thr=oscr_y_thr, y_thr_std=np.zeros_like(oscr_y_thr),
                                auc=oscr_auc, auc_std=0.0)

    # TPIR @ FPIR=10%, TPIR @ FPIR=1%, TPIR @ FPIR=0.1%, TPIR @ FPIR=0.01%
    fpir_targets = np.array([1e-1, 1e-2, 1e-3, 1e-4])
    main_fpir_op_points = {
        fpir: OSI_FPIR_OpPoint(
                fpir=fpir,
                tpir=np.interp(fpir, fpir_ascending, tpir_ascending,
                               left=0, right=tpir_ascending[-1]),
                thr=np.interp(fpir, fpir_ascending, thrs_descending,
                              left=thrs_descending[0], right=thrs_descending[-1]),
                tpir_std=0.0, thr_std=0.0
            )
        for fpir in fpir_targets
    }

    # ROC: x=FPR(FPIR) vs y=TPR/Thresholds

    #   Thrs are ascending, FPR_t is descending, TPR_t is descending
    #   ---> but ROC requires -->
    #   FPR_t is ascending, TPR_t is ascending -> Thrs must be descending

    fpr_ascending = FPR_t[::-1]
    tpr_ascending = TPR_t[::-1]
    thrs_descending = thr_stats.thrs_sorted[::-1]

    roc_x_fpr = np.linspace(0.0, 1.0, 500)
    roc_y_tpr = np.interp(roc_x_fpr, fpr_ascending, tpr_ascending,
                          left=0.0, right=1.0)
    roc_y_thr = np.interp(roc_x_fpr, fpr_ascending, thrs_descending,
                          left=thrs_descending[0], right=thrs_descending[-1])
    roc_auc = np.trapz(tpr_ascending, fpr_ascending)

    roc_curve = OSI_ROC_Curve(x_fpr=roc_x_fpr, x_fpr_std=np.zeros_like(roc_x_fpr),
                              y_tpr=roc_y_tpr, y_tpr_std=np.zeros_like(roc_y_tpr),
                              y_thr=roc_y_thr, y_thr_std=np.zeros_like(roc_y_thr),
                              auc=roc_auc, auc_std=0.0)

    # DET: x=FPR vs y=FNR(FNIR)

    #   Thrs are ascending, FPR_t is descending, FNR_t is ascending
    #   ---> but DET requires -->
    #   FPR_t is ascending, TPR_t is descending -> Thrs must be descending

    fpr_ascending = FPR_t[::-1]
    fnr_descending = FNR_t[::-1]
    thrs_descending = thr_stats.thrs_sorted[::-1]

    det_x_fpr = np.linspace(0.0, 1.0, 500)
    det_y_fnr = np.interp(det_x_fpr, fpr_ascending, fnr_descending,
                          left=fnr_descending[0],
                          right=fnr_descending[-1])
    det_y_thr = np.interp(det_x_fpr, fpr_ascending, thrs_descending,
                          left=thrs_descending[0],
                          right=thrs_descending[-1])
    det_auc = np.trapz(fnr_descending, fpr_ascending)

    det_curve = OSI_DET_Curve(x_fpr=det_x_fpr, x_fpr_std=np.zeros_like(det_x_fpr),
                              y_fnr=det_y_fnr, y_fnr_std=np.zeros_like(det_y_fnr),
                              y_thr=det_y_thr, y_thr_std=np.zeros_like(det_y_thr),
                              auc=det_auc, auc_std=0.0)

    # Equal Error Rate: FPIR ≈ FNIR
    eer_idx = np.argmin(np.abs(FPIR_t - FNIR_t))
    eer_val = (FPIR_t[eer_idx] + FNIR_t[eer_idx]) / 2
    eer_thr = thr_stats.thrs_sorted[eer_idx]

    eer = OSI_EER(val=eer_val, val_std=0.0,
                  thr=eer_thr, thr_std=0.0)

    return OSI_Metrics(oscr_curve=oscr_curve,
                       roc_curve=roc_curve,
                       det_curve=det_curve,
                       main_fpir_op_points=main_fpir_op_points,
                       eer=eer)


def _get_OSI_acc_rej_counts_per_thr(query_scores: np.ndarray,
                                    thrs_sorted: np.ndarray):
    """
    Calculate counts of accepted and rejected queries for each threshold.
    Query is accepted if its score is larger or equal to given threshold.

    Parameters:
        query_scores (np.ndarray, shape=(N_queries, N_labels), dtype=np.float32):
            Score for each query.

        thrs_sorted (np.ndarray, shape=(N_thresholds), dtype=np.float32):
            Sorted thresholds in ascending order.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            accepted_counts (np.ndarray, shape=(N_thresholds), dtype=np.int):\
                Counts of accepted queries for each threshold.

            rejected_counts (np.ndarray, shape=(N_thresholds), dtype=np.int):\
                Counts of rejected queries for each threshold.
    """

    N_queries = len(query_scores)

    # sort query scores
    # (N_queries)
    query_scores_sorted = np.sort(query_scores)

    # for each threshold, compute the insertion index
    #   in the sorted query scores array
    # this index equals the number of scores strictly less than the threshold
    # score_idx[j] = number of scores < threshold[j]
    # (N_thresholds)
    score_idx = np.searchsorted(query_scores_sorted, thrs_sorted, side="left")

    rejected_counts = score_idx
    accepted_counts = N_queries - rejected_counts

    return accepted_counts, rejected_counts


def _calc_OSI_thr_decision_stats(top1_labels: np.ndarray,
                                 top1_scores: np.ndarray,
                                 query_labels: np.ndarray,
                                 unknown_query_mask: np.ndarray):
    """
    Calculate per-threshold decision statistics.

    Parameters:
        top1_labels (np.ndarray, shape=(N_queries,), dtype=int):
            Top 1 predicted label for each query. (Excludes unknown labels).

        top1_scores (np.ndarray. shape=(N_queries,), dtype=np.float32):
            Scores of the top 1 predicted labels.

        query_labels (np.ndarray, shape=(N_labels), dtype=int): Query labels (ground truth).

        unknown_query_mask (np.ndarray, shape=(N_labels), dtype=bool):
            Boolean mask over queries. True if the query label is unknown.

    Returns:
        OSI_ThresholdDecisionStats:
            Dataclass containing per-threshold decision statistics.
    """

    # STEP 1: Preparation

    # compute query counts
    N_queries = len(query_labels)
    N_queries_unknown = np.sum(unknown_query_mask)
    N_queries_known = N_queries - N_queries_unknown

    # STEP 2: Get thresholds == all uniques scores in sorted order

    # generate thresholds
    # (N_thresholds)
    unique_scores = np.unique(top1_scores)

    # add 2 new threshold: one smaller and the other larger than all of the scores
    eps = 1e-6
    min_score = np.min(unique_scores)
    max_score = np.max(unique_scores)
    thrs_sorted = np.union1d(unique_scores, [min_score - eps, max_score + eps])

    # STEP 3: Compute query count statistics (rejected, accepted, ...) for each possible threshold

    # unknown sample count statistics

    unknown_query_scores = top1_scores[unknown_query_mask]

    unknown_accepted_counts, unknown_rejected_counts = _get_OSI_acc_rej_counts_per_thr(
        unknown_query_scores, thrs_sorted)

    # known sample count statistics

    known_query_mask = ~unknown_query_mask
    known_query_scores = top1_scores[known_query_mask]

    known_accepted_counts, known_rejected_counts = _get_OSI_acc_rej_counts_per_thr(
        known_query_scores, thrs_sorted)

    correct_query_mask = (top1_labels == query_labels)
    known_correct_query_mask = correct_query_mask & known_query_mask
    known_correct_query_scores = top1_scores[known_correct_query_mask]

    known_accepted_correct_counts, _ = _get_OSI_acc_rej_counts_per_thr(
        known_correct_query_scores, thrs_sorted)

    known_accepted_wrong_counts = known_accepted_counts - known_accepted_correct_counts

    return OSI_ThresholdDecisionStats(thrs_sorted=thrs_sorted,
                                      N_queries_known=N_queries_known,
                                      N_queries_unknown=N_queries_unknown,
                                      unknown_accepted_counts=unknown_accepted_counts,
                                      unknown_rejected_counts=unknown_rejected_counts,
                                      known_accepted_correct_counts=known_accepted_correct_counts,
                                      known_accepted_wrong_counts=known_accepted_wrong_counts,
                                      known_rejected_counts=known_rejected_counts)


def _calc_OSI_metrics(class_labels_ranked: np.ndarray,
                      class_scores_ranked: np.ndarray,
                      query_labels: np.ndarray):
    """
    Calculate open-set identification metrics.

    Parameters:
        class_labels_ranked (np.ndarray, shape=(N_queries, N_labels), dtype=int):
            Ranked labels for each query.

        class_scores_ranked (np.ndarray, shape=(N_queries, N_labels), dtype=np.float32):
            Scores of ranked labels for each query (descending order).

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

    # create boolean mask over class_labels_ranked 2D array,
    #   where True indicates that the label is known
    # (N_queries, N_labels)
    known_mask = ~np.isin(class_labels_ranked, unknown_labels)

    # remove unknown labels and their scores
    # (N_queries, N_labels) = class_labels_ranked
    # (N_queries * (N_labels - N_labels_unknown)) = class_labels_ranked[mask]  (boolean indexing flattens array)
    # (N_queries, N_labels - N_labels_unknown) = class_labels_ranked[mask].reshape(N_queries, -1)
    class_labels_ranked_k = class_labels_ranked[known_mask].reshape(
        N_queries, -1)
    class_scores_ranked_k = class_scores_ranked[known_mask].reshape(
        N_queries, -1)

    # get top 1 predicted labels and their scores
    # (N_queries)
    top1_labels = class_labels_ranked_k[:, 0]
    top1_scores = class_scores_ranked_k[:, 0]

    # STEP 4: Calculate per threshold decision query count statistics
    thr_stats = _calc_OSI_thr_decision_stats(top1_labels, top1_scores,
                                             query_labels, unknown_query_mask)

    # STEP 5: Calculate metrics from the per threshold decision query count statistics
    osi_metrics = _calc_OSI_metrics_from_thr_decision_stats(thr_stats)

    return osi_metrics
