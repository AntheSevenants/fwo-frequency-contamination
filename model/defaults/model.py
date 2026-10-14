from itertools import product

import model.constructions.sample
import model.constructions.vocab
import model.constructions.parsing
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
    a_prior: float = 0.4
    b_prior: float = 0.4
    ambiguity_prob: float = 0.5
    base_non_s: float = 0.2

    # TODO: add different groups here as well

    # ----
    # Agent age
    # ----
    agent_age_mean: int = 0
    agent_age_range: int = 200

    # ----
    # Cognition parameters
    # ----
    parsing_mode: (
        model.constructions.parsing.ExactParsing
        | model.constructions.parsing.LazyParsing
    ) = field(default_factory=lambda: model.constructions.parsing.ExactParsing())

    def __post_init__(self):
        # Initialise random number generator
        self.nprandom = np.random.default_rng(self.seed)

        if self.a_prior + self.b_prior > 1:
            raise Exception("A and B prior sum cannot exceed 1")
