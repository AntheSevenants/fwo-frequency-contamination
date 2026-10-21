from typing import Tuple
from functools import cached_property

import numpy as np
import model.enums

# Save the indices of the "contaminated" rows
contamination_idx_a = model.enums.State.get_state(
    model.enums.Construction.A, model.enums.Contamination.CONTAMINATED
)
contamination_idx_b = model.enums.State.get_state(
    model.enums.Construction.B, model.enums.Contamination.CONTAMINATED
)
contamination_indices = np.array([contamination_idx_a, contamination_idx_b], dtype=int)


class Tally:
    def __init__(
        self, base_contaminated: float, baseline_frequency: int, a_b_split: float
    ):
        # Build frequency tallies
        self.frequency: np.ndarray = np.zeros(model.enums.State._COUNT)

        baseline_frequency_a = round(a_b_split * baseline_frequency)
        baseline_frequency_b = baseline_frequency - baseline_frequency_a

        frequency_contaminated_a = round(baseline_frequency_a * base_contaminated)
        frequency_none_a = baseline_frequency_a - frequency_contaminated_a

        # Set starting frequency
        self.frequency[
            model.enums.State.get_state(
                model.enums.Construction.A, model.enums.Contamination.NONE
            )
        ] = frequency_none_a
        self.frequency[
            model.enums.State.get_state(
                model.enums.Construction.A, model.enums.Contamination.CONTAMINATED
            )
        ] = frequency_contaminated_a
        self.frequency[
            model.enums.State.get_state(
                model.enums.Construction.B, model.enums.Contamination.NONE
            )
        ] = baseline_frequency_b

    def update(self, construction: int, contamination: int):
        column_idx = model.enums.State.get_state(construction, contamination)

        # TODO: perhaps implement lateral inhibition
        self.frequency[column_idx] += 1

        # Invalidate cache
        if "__frequency__" in self.__dict__:
            del self.__dict__["__frequency__"]

        if "__contamination__" in self.__dict__:
            del self.__dict__["__contamination__"]

    def get_production_counts(
        self, construction: int
    ) -> Tuple[Tuple[int, int], np.ndarray]:
        contamination_idx = model.enums.State.get_state(
            construction, model.enums.Contamination.CONTAMINATED
        )
        regular_idx = model.enums.State.get_state(
            construction, model.enums.Contamination.NONE
        )

        indices = np.array([contamination_idx, regular_idx], dtype=int)
        return (
            model.enums.Contamination.CONTAMINATED,
            model.enums.Contamination.NONE,
        ), self.frequency[indices]

    def get_reception_counts(
        self, construction: int, contamination: int
    ) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], np.ndarray]:
        real_profile = (construction, contamination)
        confusion_profile = (
            np.abs(construction - 1),
            np.abs(contamination - 1),
        )
        real_idx = model.enums.State.get_state(*real_profile)
        confusion_idx = model.enums.State.get_state(*confusion_profile)

        indices = np.array([real_idx, confusion_idx], dtype=int)
        return (real_profile, confusion_profile), self.frequency[indices]

    @cached_property
    def __frequency__(self) -> np.ndarray:
        return self.frequency

    @cached_property
    def __contamination__(self) -> np.ndarray:
        a1, a2 = self.__frequency__[0], self.__frequency__[1]
        b1, b2 = self.__frequency__[2], self.__frequency__[3]

        contamination_shares = np.array([a2 / (a1 + a2), b2 / (b1 + b2)])

        return contamination_shares
