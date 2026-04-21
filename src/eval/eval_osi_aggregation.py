import numpy as np
from typing import List

from .metrics_osi import (
    OSI_Metrics, OSI_OSCR_Curve, OSI_ROC_Curve, OSI_DET_Curve, OSI_EER, OSI_FPIR_OpPoint)


def _aggregate_OSI_OSCR_Curve(osi_metrics_list: List[OSI_Metrics]):
    x_fpir = osi_metrics_list[0].oscr_curve.x_fpir
    y_tpir_stack = np.stack(
        [m.oscr_curve.y_tpir for m in osi_metrics_list], axis=0)
    thr_stack = np.stack(
        [m.oscr_curve.y_thr for m in osi_metrics_list], axis=0)
    auc_stack = np.array([m.oscr_curve.auc for m in osi_metrics_list])

    return OSI_OSCR_Curve(
        x_fpir=x_fpir,
        y_tpir=y_tpir_stack.mean(axis=0),
        y_thr=thr_stack.mean(axis=0),
        auc=auc_stack.mean(),
        y_tpir_std=y_tpir_stack.std(axis=0),
        y_thr_std=thr_stack.std(axis=0),
        auc_std=auc_stack.std()
    )


def _aggregate_OSI_ROC_Curve(osi_metrics_list: List[OSI_Metrics]):
    x_fpr = osi_metrics_list[0].roc_curve.x_fpr

    y_tpr_stack = np.stack(
        [m.roc_curve.y_tpr for m in osi_metrics_list], axis=0)
    thr_stack = np.stack([m.roc_curve.y_thr for m in osi_metrics_list], axis=0)
    auc_stack = np.array([m.roc_curve.auc for m in osi_metrics_list])

    return OSI_ROC_Curve(
        x_fpr=x_fpr,
        y_tpr=y_tpr_stack.mean(axis=0),
        y_thr=thr_stack.mean(axis=0),
        auc=auc_stack.mean(),
        y_tpr_std=y_tpr_stack.std(axis=0),
        y_thr_std=thr_stack.std(axis=0),
        auc_std=auc_stack.std()
    )


def _aggregate_OSI_DET_Curve(osi_metrics_list: List[OSI_Metrics]):
    x_fpr = osi_metrics_list[0].det_curve.x_fpr

    y_fnr_stack = np.stack(
        [m.det_curve.y_fnr for m in osi_metrics_list], axis=0)
    thr_stack = np.stack([m.det_curve.y_thr for m in osi_metrics_list], axis=0)
    auc_stack = np.array([m.det_curve.auc for m in osi_metrics_list])

    return OSI_DET_Curve(
        x_fpr=x_fpr,
        y_fnr=y_fnr_stack.mean(axis=0),
        y_thr=thr_stack.mean(axis=0),
        auc=auc_stack.mean(),
        y_fnr_std=y_fnr_stack.std(axis=0),
        y_thr_std=thr_stack.std(axis=0),
        auc_std=auc_stack.std()
    )


def _aggregate_OSI_EER(osi_metrics_list: List[OSI_Metrics]):
    val_stack = np.array([m.eer.val for m in osi_metrics_list])
    thr_stack = np.array([m.eer.thr for m in osi_metrics_list])

    return OSI_EER(
        val=val_stack.mean(),
        thr=thr_stack.mean(),
        val_std=val_stack.std(),
        thr_std=thr_stack.std()
    )


def _aggregate_OSI_main_FPIR_OpPoints(osi_metrics_list: List[OSI_Metrics]):
    # FPIR key values are the same for each run
    fpir_keys = osi_metrics_list[0].main_fpir_op_points.keys()

    res_dict = {}

    for fpir in fpir_keys:
        # stack TPIR values from each run
        tpir_stack = np.array([
            m.main_fpir_op_points[fpir].tpir
            for m in osi_metrics_list
        ])

        # stack threshold values for each run
        thr_stack = np.array([
            m.main_fpir_op_points[fpir].thr
            for m in osi_metrics_list
        ])

        # create operation point
        res_dict[fpir] = OSI_FPIR_OpPoint(
            fpir=fpir,
            tpir=tpir_stack.mean(),
            thr=thr_stack.mean(),
            tpir_std=tpir_stack.std(),
            thr_std=thr_stack.std(),
        )

    return res_dict


def _aggregate_OSI_metrics(osi_metrics_list: List[OSI_Metrics]):
    """
    Aggregates list of computed OSI Metrics for different random known/unknown splits.
    For each metric value (except fixed ones) calcualte mean and std and store it in OSI_Metrics.

    Parameters:
        osi_metrics_list (List[OSI_Metrics]):
            List of computed OSI Metrics for different random known/unknown splits.

    Returns:
        OSI_Metrics:
            Aggregated OSI Metrics.
    """

    return OSI_Metrics(
        oscr_curve=_aggregate_OSI_OSCR_Curve(osi_metrics_list),
        roc_curve=_aggregate_OSI_ROC_Curve(osi_metrics_list),
        det_curve=_aggregate_OSI_DET_Curve(osi_metrics_list),
        main_fpir_op_points=_aggregate_OSI_main_FPIR_OpPoints(osi_metrics_list),
        eer=_aggregate_OSI_EER(osi_metrics_list)
    )
