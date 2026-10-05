import copy
import numpy as np

from dataclasses import dataclass, asdict, field
from typing import List, Optional

import model.defaults.model
import model.constructions.vocabulary
import model.enums


@dataclass
class Attributes:
    model_params: model.defaults.model.Parameters
    vocabulary: model.constructions.vocabulary.Vocabulary

    max_age: int | None
    age: int = 0  # How old is the agent currently?

    def __post_init__(self):
        # We need the parent model parameters in order to be able to initialise
        # the probabilities and starting probabilities and such
        if self.model_params is None:
            raise ValueError("Model parameters cannot be None")

        # Make a deepcopy of the vocabulary
        self.vocabulary = copy.deepcopy(self.vocabulary)
