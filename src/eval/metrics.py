import numpy as np
from dataclasses import dataclass, asdict
import json
import re


@dataclass(frozen=True)
class OSI_OSCR_Curve:
    """ OSCR: TPIR x FPIR """

    x_fpir: np.ndarray
    y_tpir: np.ndarray
    y_thr: np.ndarray
    auc: np.float32

    x_fpir_std: np.ndarray
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

    x_fpr_std: np.ndarray
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

    x_fpr_std: np.ndarray
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

    def to_json(self) -> str:
        """
        Returns string JSON representation of IdnetificationMetrics.
        Ideal for saving to file.
        """
        d = asdict(self)
        d = dict_convert_numpy_types_to_python(d)
        return json.dumps(d, indent=4)

    def to_json_compact(self) -> str:
        """
        Returns compact string JSON representation of IdnetificationMetrics.
        Only single scalar metrics are included.
        Ideal for printing to stdout.
        """
        d = asdict(self)
        d = dict_remove_arr_fields(d)
        d = dict_convert_numpy_types_to_python(d)
        d = dict_round_floats(d, decimals=4)
        return json_dumps_compact(d, indent=4)


def dict_remove_arr_fields(d: dict) -> dict:
    """
    Iterate over all fields in dict containing nested dicts
    and remove all values with list or np.ndarray types.
    """
    res_d = {}
    for key, value in d.items():
        if isinstance(value, dict):
            # remove arrays from nested dict
            res_d[key] = dict_remove_arr_fields(value)
        elif isinstance(value, (list, np.ndarray)):
            # skipping lists and np.ndarray values
            continue
        else:
            # other values are kept intact
            res_d[key] = value
    return res_d


def dict_convert_numpy_types_to_python(data):
    """
    Iterate over all fields in dict containing nested dicts
    and round all float values to specified decimals.
    """
    if isinstance(data, dict):
        # convert each dict value
        return {key: dict_convert_numpy_types_to_python(value)
                for key, value in data.items()}
    elif isinstance(data, list):
        # convert each list element
        return [dict_convert_numpy_types_to_python(item)
                for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    else:
        return data


def dict_round_floats(data, decimals=4):
    """
    Iterate over all fields in dict containing nested dicts
    and round all float values to specified decimals.

    Does not work with numpy types, so call dict_convert_numpy_types_to_python first.
    """
    if isinstance(data, dict):
        # call round_floats_in_dict on each dict field value
        return {key: dict_round_floats(value, decimals)
                for key, value in data.items()}
    elif isinstance(data, list):
        # call round_floats_in_dict on each list element
        return [dict_round_floats(item, decimals)
                for item in data]
    elif isinstance(data, float):
        return round(data, decimals)
    else:
        return data


def json_dumps_compact(data, indent=4):
    """
    Wrapper to json.dumps function, that 'copresses' leaf objects to one line.
    """

    # run json dumps
    dump = json.dumps(data, indent=indent)

    # replace any multiline { ... } that doesnt contain nested { ... } or [ ... ]
    # with inline version (no '\n')
    compact_dump = re.sub(
        r'\{[^}{]*?\n\s*("[^"]*"\s*:\s*[^}{[\]]*?,?\s*\n\s*)*"[^"]*"\s*:\s*[^}{[\]]*?\n\s*\}',
        lambda m: re.sub(r'\s+', ' ', m.group(0)),
        dump
    )

    return compact_dump
