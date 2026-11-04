from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Tuple, Union

import numpy as np
import pandas as pd


def derive_outcome(contamination_share: float):
    if np.isclose(contamination_share, 0):
        return "avoidance"
    elif np.isclose(contamination_share, 1):
        return "contamination"
    else:
        return "unclear"


def get_base_contaminated(base_contaminated: int | float | List[float]) -> float:
    if type(base_contaminated) == float:
        return base_contaminated
    elif type(base_contaminated) == int:
        return float(base_contaminated)
    elif type(base_contaminated) == list:
        return float(base_contaminated[0])
    else:
        raise TypeError("Base contaminated is of incorrect type")


posthoc_operations: Dict[str, Callable[[pd.Series, Dict[str, Any]], Any]] = {
    "a_prob": lambda row, combination_data: row["a_b_split"],
    "b_prob": lambda row, combination_data: 1 - row["a_b_split"],
    "a_cont_prod_prob": lambda row, combination_data: row["a_b_split"]
    * get_base_contaminated(row["base_contaminated"]),
    "contamination_mean": lambda row, combination_data: combination_data[
        "contamination_mean"
    ]["mean"][-1][0],
    "contamination_q1": lambda row, combination_data: combination_data[
        "contamination_mean"
    ]["q1"][-1][0],
    "contamination_q3": lambda row, combination_data: combination_data[
        "contamination_mean"
    ]["q3"][-1][0],
    "expectation": lambda row, combination_data: (
        "contamination"
        if row["a_b_split"] * get_base_contaminated(row["base_contaminated"])
        >= (1 - row["a_b_split"])
        else "avoidance"
    ),
    "outcome": lambda row, combination_data: derive_outcome(
        combination_data["contamination_mean"]["mean"][-1][0]
    ),
}
