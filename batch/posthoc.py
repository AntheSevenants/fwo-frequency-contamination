from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Tuple, Union

import numpy as np
import pandas as pd


def derive_outcome(contamination_share: float):
    if contamination_share <= 0.05:
        return "avoidance"
    elif contamination_share >= 0.95:
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
    "mean_cont": lambda row, combination_data: combination_data["contamination_median"][
        "median"
    ][-1][0],
    "q1_cont": lambda row, combination_data: combination_data["contamination_median"][
        "q1"
    ][-1][0],
    "q3_cont": lambda row, combination_data: combination_data["contamination_median"][
        "q3"
    ][-1][0],
    "expectation": lambda row, combination_data: (
        "contamination"
        if row["a_b_split"] * get_base_contaminated(row["base_contaminated"])
        >= (1 - row["a_b_split"])
        else "avoidance"
    ),
    "outcome": lambda row, combination_data: derive_outcome(
        combination_data["contamination_median"]["median"][-1][0]
    ),
}
