import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class OSI_OSCR_Curve:
    """ OSCR: TPIR x FPIR """

    x_fpir: np.ndarray

    y_tpir: np.ndarray
    y_thr: np.ndarray
    auc: np.float32

    y_tpir_std: np.ndarray
    y_thr_std: np.ndarray
    auc_std: np.float32


@dataclass(frozen=True)
class OSI_ROC_Curve:
    """ ROC: TPR x FPR(FPIR) """

    x_fpr: np.ndarray

    y_tpr: np.ndarray
    y_thr: np.ndarray
    auc: np.float32

    y_tpr_std: np.ndarray
    y_thr_std: np.ndarray
    auc_std: np.float32


@dataclass(frozen=True)
class OSI_DET_Curve:
    """ DET: FNR(FNIR) x FPR """

    x_fpr: np.ndarray

    y_fnr: np.ndarray
    y_thr: np.ndarray
    auc: np.float32

    y_fnr_std: np.ndarray
    y_thr_std: np.ndarray
    auc_std: np.float32


@dataclass(frozen=True)
class OSI_FPIR_OpPoint:
    """ TPIR and Threshold at specific FPIR operating points """

    fpir: np.float32

    tpir: np.float32
    thr: np.float32

    tpir_std: np.float32
    thr_std: np.float32


@dataclass(frozen=True)
class OSI_EER:
    """
    EER = Equal Error Rate

    Point on DET curve, where FPIR ≈ FNIR
    """

    val: np.float32
    thr: np.float32

    val_std: np.float32
    thr_std: np.float32


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
