from itertools import product

import model.constructions.sample
import numpy as np

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Type, Any, Tuple

# Mapping of parameter names to their enum classes
PARAMETER_ENUM_MAPPING: Dict[str, Type] = {
    # nothing yet
}


@dataclass
class Parameters:
    # ----
    # Model housekeeping
    # ----
    num_agents: int = 10
    seed: Optional[int] = None

    # After how many steps do we collect data?
    datacollector_step_size: int = 1

    # ----
    # Construction info
    # ----
    a_b_split: float = 0.4

    ambiguity_prob: float | List[float] = 0.5  # dim 0 = CA, dim 1 = CB
    # initial contamination
    base_contaminated: float | List[float] = 0.0  # dim 0 = group A, dim 1 = group B
    learning_rate: float = 0.01  # how fast do beliefs change?

    # ----
    # Group information
    # ----
    a_group_share: float = 1
    homophily: List[float] | None = None

    # ----
    # Agent age
    # ----
    agent_age_mean: int = 0
    agent_age_range: int = 200

    def __post_init__(self):
        # Initialise random number generator
        self.nprandom = np.random.default_rng(self.seed)

        # Is there a group distinction?
        self.group_distinction = self.a_group_share not in [0, 1]

        # Split into separate ambiguity probabilities
        if type(self.ambiguity_prob) == float:
            self.ambiguity_probs = [self.ambiguity_prob, self.ambiguity_prob]
        elif type(self.ambiguity_prob) == list:
            if len(self.ambiguity_prob) != 2:
                raise ValueError("ambiguity_prob should be length 2 if a list")

            self.ambiguity_probs = self.ambiguity_prob
        else:
            raise TypeError("ambiguity_prob should be float or List[float] of size 2")

        # If homophily is set and a_group_share is 0 or 1, the parameter does not make sense
        if self.homophily is not None and self.a_group_share in [0, 1]:
            raise ValueError("Homophily cannot be set if group share is 0 or 1")

        if self.homophily is not None:
            if len(self.homophily) != 2:
                raise ValueError("Homophily should be a list of length 2")
