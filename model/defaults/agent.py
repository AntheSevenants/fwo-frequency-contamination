import copy
import numpy as np

from dataclasses import dataclass, asdict, field
from typing import List, Optional

import model.defaults.model
import model.constructions.tally
import model.enums


@dataclass
class Attributes:
    model_params: model.defaults.model.Parameters

    group: int
    base_contaminated: float

    max_age: int | None
    age: int = 0  # How old is the agent currently?

    def __post_init__(self):
        # We need the parent model parameters in order to be able to initialise
        # the probabilities and starting probabilities and such
        if self.model_params is None:
            raise ValueError("Model parameters cannot be None")

        # Make a deepcopy of the vocabulary
        self.tally = model.constructions.tally.Tally(self.base_contaminated)

        # Prepare for construction choice
        self.choices = [model.enums.Construction.A, model.enums.Construction.B, None]
        probabilities = [self.model_params.a_prior, self.model_params.b_prior]
        probabilities.append(1 - np.sum(probabilities))

        self.probabilities = np.array(probabilities)

    def get_construction(self):
        return self.model_params.nprandom.choice(self.choices, p=self.probabilities)

    def get_ambiguity_prob(self, construction: int):
        return self.model_params.ambiguity_prob
