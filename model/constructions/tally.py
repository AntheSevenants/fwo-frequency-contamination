from typing import Tuple
from functools import cached_property

import numpy as np
import model.enums
import model.constructions.vocabulary

# Save the indices of the "contaminated" rows
contamination_idx_a = model.enums.State.get_state(
    model.enums.Construction.A, model.enums.Contamination.CONTAMINATED
)
contamination_idx_b = model.enums.State.get_state(
    model.enums.Construction.B, model.enums.Contamination.CONTAMINATED
)
contamination_indices = np.array([contamination_idx_a, contamination_idx_b], dtype=int)


class Tally:
    def __init__(self, vocabulary: model.constructions.vocabulary.Vocabulary):
        # Build frequency tallies
        self.frequency: np.ndarray = np.zeros(
            (vocabulary.num_words, model.enums.State._COUNT)
        )

        regular_indices = np.array(
            [
                model.enums.State.get_state(
                    model.enums.Construction.A, model.enums.Contamination.NONE
                ),
                model.enums.State.get_state(
                    model.enums.Construction.B, model.enums.Contamination.NONE
                ),
            ]
        )
        # Set starting frequency to one for all "regular" occurrences
        self.frequency[:, regular_indices] = 1

    def update(self, word_index: int, construction: int, contamination: int):
        column_idx = model.enums.State.get_state(construction, contamination)

        # TODO: perhaps implement lateral inhibition
        self.frequency[word_index, column_idx] += 1

        # Invalidate cache
        if "__frequency__" in self.__dict__:
            del self.__dict__["__frequency__"]

        if "__contamination__" in self.__dict__:
            del self.__dict__["__contamination__"]

    def get_production_counts(
        self, word_index: int, construction: int
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
        ), self.frequency[word_index, indices]

    def get_reception_counts(
        self, word_index: int, construction: int, contamination: int
    ) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], np.ndarray]:
        real_profile = (construction, contamination)
        confusion_profile = (
            np.abs(construction - 1),
            np.abs(contamination - 1),
        )
        real_idx = model.enums.State.get_state(*real_profile)
        confusion_idx = model.enums.State.get_state(*confusion_profile)

        indices = np.array([real_idx, confusion_idx], dtype=int)
        return (real_profile, confusion_profile), self.frequency[word_index, indices]

    @cached_property
    def __frequency__(self) -> np.ndarray:
        return self.frequency

    @property
    def __frequency_mean__(self) -> np.ndarray:
        return np.mean(self.__frequency__, axis=0)

    @property
    def __frequency_median__(self) -> np.ndarray:
        return np.median(self.__frequency__, axis=0)

    @cached_property
    def __contamination__(self) -> np.ndarray:
        matrix = self.__frequency__

        # Reshape to (rows, number_of_pairs, 2)
        reshaped = matrix.reshape(matrix.shape[0], -1, 2)

        # Calculate the sum of each pair along the last axis
        pair_sums = reshaped.sum(axis=2, keepdims=True)

        # Divide the reshaped matrix by the sums,
        # reshape back to original
        result = (reshaped / pair_sums).reshape(matrix.shape)

        # Now, only get the contamination values
        contamination_shares = result[:, contamination_indices]

        return contamination_shares

    @property
    def __contamination_mean__(self) -> np.ndarray:
        return np.mean(self.__contamination__, axis=0)

    @property
    def __contamination_median__(self) -> np.ndarray:
        return np.median(self.__contamination__, axis=0)
