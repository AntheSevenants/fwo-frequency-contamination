from typing import Tuple

import numpy as np
import model.enums
import model.constructions.vocabulary


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
