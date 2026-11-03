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


class Beliefs:
    def __init__(
        self,
        a_prior: float,
        contamination_prior_given_a: float,
        contamination_prior_given_b: float,
        learning_rate: float,
    ):
        """Initialise the beliefs system for this agent based on the given priors and posteriors

        Args:
            a_prior (float): Probability of occurrence for A. B = 1 - A
            contamination_prior_given_a (float): Initial contamination prob for A.
            contamination_prior_given_b (float): Initial contamination prob for B.
            learning_rate (float): The learning rate (>0 and <= 1). Higher means faster learning; lower means more stability.
        """

        self.learning_rate = learning_rate

        # P(A) and P(B)
        self.construction_p = np.array([a_prior, 1.0 - a_prior])

        # P(Contamination | Construction)
        self.contamination_p = np.array(
            [
                [1.0 - contamination_prior_given_a, contamination_prior_given_a],
                [1.0 - contamination_prior_given_b, contamination_prior_given_b],
            ]
        )

    def update(self, construction: int, contamination: int):
        """Update the frequency count of a construction with/without contamination.

        Args:
            construction (int): Either of the two constructions: model.enums.Construction.A | model.enums.Construction.B
            contamination (int): Either of the two contamination states: model.enums.contamination.CONTAMINATED | model.enums.contamination.NONE
        """

        # === (1) Construction probabilities ===
        x_obs = np.zeros(2)
        x_obs[construction] = 1

        # Apply Estimated Moving Average: p = p + lambda * (x - p)
        self.construction_p += self.learning_rate * (x_obs - self.construction_p)

        # === (2) Conditional contamination probabilities ===
        x_cont_obs = np.zeros(2)
        x_cont_obs[contamination] = 1

        # Apply Estimated Moving Average to the row corresponding to the heard construction
        # p_cont = p_cont + alpha * (x_cont_obs - p_cont)
        self.contamination_p[construction] += self.learning_rate * (
            x_cont_obs - self.contamination_p[construction]
        )

        # Invalidate cache
        if "__contamination__" in self.__dict__:
            del self.__dict__["__contamination__"]

    def get_a_b_split_introspection(self) -> np.ndarray:
        """Get the total probabilities for constructions A and B;

        Returns:
            np.ndarray: np array with probs for constructions A and B respectively
        """

        return self.construction_p

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

        return float(self.contamination_p[construction, contamination])

    def get_construction_prob_introspection(self, construction: int) -> float:
        """Get the probability of a given construction

        Args:
            construction (int): Either of the two constructions: model.enums.Construction.A | model.enums.Construction.B

        Returns:
            float: The probability of the given construction
        """

        return float(self.construction_p[construction])

    @cached_property
    def __contamination__(self) -> np.ndarray:
        """Contamination shares for both constructions

        Returns:
            np.ndarray: Degree of contamination for each construction, as a numpy array
        """

        return self.contamination_p[:, model.enums.Contamination.CONTAMINATED]
