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
        """Initialise the frequency tally for this agent based on the A/B/ split, base contamination and baseline frequency

        Args:
            base_contaminated (float): Ratio of Construction A is contaminated from the start?
            baseline_frequency (int): Total frequency count of entire tally
            a_b_split (float): Ratio of occurrence of Construction A. B = (1 - A)
        """

        # Build frequency tallies
        self.frequency: np.ndarray = np.zeros(model.enums.State._COUNT)

        # If A/B = 0.4 and baseline frequency = 100, A = 40 and B = 60
        baseline_frequency_a = round(a_b_split * baseline_frequency)
        baseline_frequency_b = baseline_frequency - baseline_frequency_a

        # If base contaminated = 0.2 and A = 40, then freq contaminated = 8
        frequency_contaminated_a = round(baseline_frequency_a * base_contaminated)
        frequency_none_a = baseline_frequency_a - frequency_contaminated_a

        # Encode these frequencies
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
        """Update the frequency count of a construction with/without contamination.

        Args:
            construction (int): Either of the two constructions: model.enums.Construction.A | model.enums.Construction.B
            contamination (int): Either of the two contamination states: model.enums.contamination.CONTAMINATED | model.enums.contamination.NONE
        """

        # First, derive the correct column index from the identity
        column_idx = model.enums.State.get_state(construction, contamination)

        # TODO: perhaps implement lateral inhibition
        # Increase frequency count for this identity
        self.frequency[column_idx] += 1

        # Invalidate cache
        if "__frequency__" in self.__dict__:
            del self.__dict__["__frequency__"]

        if "__contamination__" in self.__dict__:
            del self.__dict__["__contamination__"]

    def get_construction_counts(
        self, construction: int
    ) -> Tuple[Tuple[int, int], np.ndarray]:
        """Get the uncontaminated/contaminated counts for a specific construction

        Args:
            construction (int): Either of the two constructions: model.enums.Construction.A | model.enums.Construction.B

        Returns:
            Tuple[Tuple[int, int], np.ndarray]: ((Contamination.NONE, Contamination.CONTAMINATED), [#none, #contaminated])
        """

        regular_idx = model.enums.State.get_state(
            construction, model.enums.Contamination.NONE
        )
        contamination_idx = model.enums.State.get_state(
            construction, model.enums.Contamination.CONTAMINATED
        )

        indices = np.array([regular_idx, contamination_idx], dtype=int)
        return (
            model.enums.Contamination.NONE,
            model.enums.Contamination.CONTAMINATED,
        ), self.frequency[indices]

    def get_a_b_split_introspection(self) -> np.ndarray:
        """Get the total counts for construction A and B;

        Returns:
            np.ndarray: np array with counts for construction A and B respectively
        """

        sums = self.frequency.reshape(2, 2).sum(axis=1)
        ratios = sums / sums.sum()

        return ratios

    def get_contamination_prob_for_construction_introspection(
        self, construction: int, contamination: int
    ) -> float:
        """Get the probability of a contamination state given a construction

        Args:
            construction (int): Either of the two constructions: model.enums.Construction.A | model.enums.Construction.B
            contamination (int): Either of the two contamination states: model.enums.contamination.CONTAMINATED | model.enums.contamination.NONE

        Returns:
            float: The probability of the given contamination state, given the supplied construction
        """

        # First, get the uncontaminated/contaminated count split
        # Profile 0 = without contamination
        # Profile 1 = with contamination
        profiles, counts = self.get_construction_counts(construction)

        # (to be sure that I get the right profile, I actually should get the index of
        #  the value of the contamination profile (0/1), and use that to index the count)
        # (... but that would waste cycles, so I'm just hardcoding this because I know
        #  the order will be correct)

        # Then, get the ratio of the required contamination state given the full count
        return float(counts[contamination] / counts.sum())

    def get_construction_prob_introspection(self, construction: int) -> float:
        """Get the probability of a given construction

        Args:
            construction (int): Either of the two constructions: model.enums.Construction.A | model.enums.Construction.B

        Returns:
            float: The probability of the given construction
        """

        return float(self.get_a_b_split_introspection()[construction])

    @cached_property
    def __frequency__(self) -> np.ndarray:
        """The frequency matrix kept internally

        Returns:
            np.ndarray: The frequency matrix
        """

        return self.frequency

    @cached_property
    def __contamination__(self) -> np.ndarray:
        """Contamination shares for both constructions

        Returns:
            np.ndarray: Degree of contamination for each construction, as a numpy array
        """

        a1, a2 = self.__frequency__[0], self.__frequency__[1]
        b1, b2 = self.__frequency__[2], self.__frequency__[3]

        contamination_shares = np.array([a2 / (a1 + a2), b2 / (b1 + b2)])

        return contamination_shares
