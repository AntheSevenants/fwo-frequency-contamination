import numpy as np

import model.enums

from model.enums import to_dict
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from model.model import ContaminationModel


class CommunicationResult:
    FAILURE_CONTAMINATION = 0
    SUCCESS = 1


class CommunicativeSuccess:
    def __init__(self, contamination_model: "ContaminationModel"):
        self.contamination_model = contamination_model

    def reset(self):
        self.outcomes = np.zeros(
            (
                len(list(to_dict(model.enums.Group).values())),
                model.enums.State._COUNT,  # intention
                model.enums.State._COUNT,  # outcome
            )
        )

    def register_communication_outcome(
        self, group: int, intended_state: int, heard_state: int
    ):
        self.outcomes[group, intended_state, heard_state] += 1

    def get_confusion_matrix_by_group(self, group: int) -> np.ndarray:
        return self.outcomes[group]

    def get_confusion_matrix_normalised(self, weights: List[float]):
        if self.outcomes.shape[0] != len(weights):
            raise ValueError(
                f"Weights should have same length as number of groups: {self.outcomes.shape[0]}"
            )

        return np.average(self.outcomes, axis=0, weights=weights)

    def get_percentages_per_ctx(self, group: int) -> np.ndarray:
        counts = self.outcomes[group]

        with np.errstate(divide="ignore", invalid="ignore"):
            shares_per_ctx = np.true_divide(counts, np.sum(counts))

        return shares_per_ctx
