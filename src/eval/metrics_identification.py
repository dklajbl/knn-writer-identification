import numpy as np
from dataclasses import dataclass, asdict
import json
import re

from .metrics_csi import CSI_Metrics
from .metrics_osi import OSI_Metrics


@dataclass
class IdentificationMetrics:
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
        d = _dict_convert_numpy_types_to_python(d)
        return json.dumps(d, indent=4)

    def to_json_compact(self) -> str:
        """
        Returns compact string JSON representation of IdnetificationMetrics.
        Only single scalar metrics are included.
        Ideal for printing to stdout.
        """
        d = asdict(self)
        d = _dict_remove_arr_fields(d)
        d = _dict_convert_numpy_types_to_python(d)
        d = _dict_round_floats(d, decimals=4)
        return _json_dumps_compact(d, indent=4)


def _dict_remove_arr_fields(d: dict) -> dict:
    """
    Iterate over all fields in dict containing nested dicts
    and remove all values with list or np.ndarray types.
    """
    res_d = {}
    for key, value in d.items():
        if isinstance(value, dict):
            # remove arrays from nested dict
            res_d[key] = _dict_remove_arr_fields(value)
        elif isinstance(value, (list, np.ndarray)):
            # skipping lists and np.ndarray values
            continue
        else:
            # other values are kept intact
            res_d[key] = value
    return res_d


def _dict_convert_numpy_types_to_python(data):
    """
    Iterate over all fields in dict containing nested dicts
    and round all float values to specified decimals.
    """
    if isinstance(data, dict):
        # convert each dict value
        return {key: _dict_convert_numpy_types_to_python(value)
                for key, value in data.items()}
    elif isinstance(data, list):
        # convert each list element
        return [_dict_convert_numpy_types_to_python(item)
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


def _dict_round_floats(data, decimals=4):
    """
    Iterate over all fields in dict containing nested dicts
    and round all float values to specified decimals.

    Does not work with numpy types, so call _dict_convert_numpy_types_to_python first.
    """
    if isinstance(data, dict):
        # call round_floats_in_dict on each dict field value
        return {key: _dict_round_floats(value, decimals)
                for key, value in data.items()}
    elif isinstance(data, list):
        # call round_floats_in_dict on each list element
        return [_dict_round_floats(item, decimals)
                for item in data]
    elif isinstance(data, float):
        return round(data, decimals)
    else:
        return data


def _json_dumps_compact(data, indent=4):
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
