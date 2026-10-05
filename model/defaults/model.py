from itertools import product

import model.constructions.sample
import model.constructions.vocabulary
import numpy as np

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Type, Any, Tuple


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
    # Vocabulary
    # ----
    num_words: int = 100

    sampling_type: (
        model.constructions.sample.ZipfianSampling
        | model.constructions.sample.ExponentialSampling
        | model.constructions.sample.LinearSampling
    ) = field(default_factory=lambda: model.constructions.sample.ZipfianSampling())
    priors_enabled: bool = True

    # ----
    # Agent age
    # ----
    agent_age_mean: int = 0
    agent_age_range: int = 200

    # ----
    # Cognition parameters
    # ----

    def __post_init__(self):
        # Initialise random number generator
        self.nprandom = np.random.default_rng(self.seed)

        # Get the priors for the chosen sampling type
        true_ranks, self.priors = self.sampling_type.get_priors(self.nprandom)

        self.vocabulary = model.constructions.vocabulary.Vocabulary(
            self.num_words, self.priors.tolist(), self.priors_enabled
        )
